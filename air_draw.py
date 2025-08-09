import cv2
import time
import math
import numpy as np
from collections import deque, Counter

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("mediapipe not installed. Run: pip install mediapipe")

# =========================
# Config
# =========================
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Brush
BRUSH_MIN, BRUSH_MAX = 3, 40
DRAW_THICKNESS_INIT = 8
ERASE_THICKNESS = 40

# GUI
SWATCH_SIZE = (80, 60)        # w, h
SWATCH_PADDING = 20
TOOLBAR_Y = 10
ALPHA_OVERLAY = 0.60

# Smoothing / debouncing
GESTURE_SMOOTH_N = 6          # frames used for majority vote
EVENT_COOLDOWN_S = 1.2        # for save/undo/clear

# Colors (BGR)
COLOR_SWATCHES = [
    ("White", (255, 255, 255)),
    ("Red",   (0,   0, 255)),
    ("Green", (0, 255,   0)),
    ("Blue",  (255, 0,   0)),
    ("Yellow",(0, 255, 255)),
    ("Purple",(255, 0, 255)),
    ("Black", (0,   0,   0)),
]

# =========================
# Utilities
# =========================
def euclid(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def most_common(items):
    if not items: return None
    return Counter(items).most_common(1)[0][0]

def bbox_of_landmarks(lm):
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    return min(xs), min(ys), max(xs), max(ys)

def fingers_up_flags(lm, handedness_label, img_h):
    """
    Robust finger-up detection.
    lm is list of (x,y) pixels for 21 hand landmarks.
    Returns list [thumb, index, middle, ring, pinky] booleans.
    """
    # Helpful indices
    TIP = { "thumb":4, "index":8, "middle":12, "ring":16, "pinky":20 }
    PIP = { "thumb":3, "index":6, "middle":10, "ring":14, "pinky":18 }
    MCP = { "thumb":2, "index":5, "middle":9, "ring":13, "pinky":17 }

    margin = int(0.02 * img_h)  # small vertical tolerance

    up = [False]*5
    # Thumb: use x-direction depending on handedness
    # Right hand: thumb extends to the RIGHT (tip.x > ip.x)
    # Left hand:  thumb extends to the LEFT  (tip.x < ip.x)
    if handedness_label == "Right":
        up[0] = (lm[TIP["thumb"]][0] - lm[PIP["thumb"]][0]) > 18
    else:
        up[0] = (lm[PIP["thumb"]][0] - lm[TIP["thumb"]][0]) > 18

    # Other fingers: tip above PIP and PIP above MCP
    for i, finger in enumerate(["index", "middle", "ring", "pinky"], start=1):
        up[i] = (lm[TIP[finger]][1] < lm[PIP[finger]][1] - margin) and (lm[PIP[finger]][1] < lm[MCP[finger]][1] - margin)

    return up

def palm_width(lm):
    # Use distance between index MCP (5) and pinky MCP (17) as palm width
    return euclid(lm[5], lm[17]) + 1e-6

def pinch(lm, scale):
    # Thumb tip (4) and Index tip (8) distance normalized by palm width
    d = euclid(lm[4], lm[8])
    return d < 0.28 * scale  # normalized threshold

def is_ok_sign(lm, ups, scale):
    # OK sign: pinch + at least 2 of middle/ring/pinky extended
    others_up = sum(ups[2:])  # middle, ring, pinky
    return pinch(lm, scale) and others_up >= 2

def is_pinch_resize(lm, ups, scale):
    # Pinch for resize: pinch + other three down (not open palm / OK)
    others_up = sum(ups[2:])  # middle, ring, pinky
    return pinch(lm, scale) and others_up == 0

def is_fist(ups):
    return sum(ups) == 0

def is_open_palm(ups):
    return sum(ups) == 5

def is_index_only(ups):
    return ups[1] and not any([ups[0], ups[2], ups[3], ups[4]])

def is_index_middle(ups):
    return ups[1] and ups[2] and not any([ups[0], ups[3], ups[4]])

def is_index_middle_ring(ups):
    return ups[1] and ups[2] and ups[3] and not any([ups[0], ups[4]])

def is_thumbs_up(lm, ups, scale):
    # Thumb extended; others down; thumb tip clearly above wrist
    if not ups[0] or any(ups[1:]): return False
    wrist = lm[0]
    return (wrist[1] - lm[4][1]) > 0.20 * scale  # thumb tip higher than wrist

def is_thumbs_down(lm, ups, scale):
    # Thumb extended; others down; thumb tip clearly below wrist
    if not ups[0] or any(ups[1:]): return False
    wrist = lm[0]
    return (lm[4][1] - wrist[1]) > 0.20 * scale  # thumb tip lower than wrist

# =========================
# GUI helpers
# =========================
def draw_toolbar(frame, current_color, mode, brush_size):
    x = SWATCH_PADDING
    for name, bgr in COLOR_SWATCHES:
        w, h = SWATCH_SIZE
        rect = (x, TOOLBAR_Y, x+w, TOOLBAR_Y+h)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), bgr, thickness=-1)
        if mode == "draw" and bgr == current_color:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255,255,255), 2)
        cv2.putText(frame, name, (rect[0]+6, rect[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(frame, name, (rect[0]+6, rect[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        x += w + SWATCH_PADDING

    # Eraser tile
    w, h = SWATCH_SIZE
    rect = (x, TOOLBAR_Y, x+w, TOOLBAR_Y+h)
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (40,40,40), thickness=-1)
    cv2.putText(frame, "ERASE", (rect[0]+8, rect[1]+h//2+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(frame, "ERASE", (rect[0]+8, rect[1]+h//2+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    if mode == "erase":
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255,255,255), 2)

    # Mode chip + brush size
    mode_text = {"draw":"Draw", "select":"Select", "erase":"Eraser", "pause":"Pause"}[mode]
    chip = f"Mode: {mode_text} | Brush: {brush_size}px"
    cv2.rectangle(frame, (20, TOOLBAR_Y + h + 30), (520, TOOLBAR_Y + h + 70), (0,0,0), -1)
    cv2.putText(frame, chip, (30, TOOLBAR_Y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

def color_from_toolbar(point):
    x = SWATCH_PADDING
    for _, bgr in COLOR_SWATCHES:
        w, h = SWATCH_SIZE
        rect = (x, TOOLBAR_Y, x+w, TOOLBAR_Y+h)
        if rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]:
            return ("color", bgr)
        x += w + SWATCH_PADDING
    # Eraser tile
    w, h = SWATCH_SIZE
    rect = (x, TOOLBAR_Y, x+w, TOOLBAR_Y+h)
    if rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]:
        return ("eraser", None)
    return (None, None)

def blend_layers(base_bgr, draw_bgr, alpha=0.6):
    mask = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2GRAY)
    mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    blended = cv2.addWeighted(base_bgr, 1 - alpha, draw_bgr, alpha, 0)
    mask_inv = cv2.bitwise_not(mask_bin)
    bg = cv2.bitwise_and(base_bgr, base_bgr, mask=mask_inv)
    fg = cv2.bitwise_and(blended, blended, mask=mask_bin)
    return cv2.add(bg, fg)

# =========================
# Stroke store (for undo)
# =========================
class Stroke:
    def __init__(self, color, thickness, erase=False):
        self.color = color
        self.thickness = thickness
        self.points = []
        self.erase = erase

def render_strokes(strokes, h, w):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for s in strokes:
        if len(s.points) < 2:
            continue
        color = (0,0,0) if s.erase else s.color
        for i in range(1, len(s.points)):
            cv2.line(canvas, s.points[i-1], s.points[i], color, s.thickness, cv2.LINE_AA)
    return canvas

# =========================
# Main
# =========================
def main():
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or CAM_INDEX.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)


    strokes = []
    current_stroke = None
    drawing_layer = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    current_color = (0, 0, 255)     # red
    brush_size = DRAW_THICKNESS_INIT
    mode = "pause"                  # draw | select | erase | pause

    # Gesture smoothing / cooldowns
    gesture_hist = deque(maxlen=GESTURE_SMOOTH_N)
    last_event = {"save":0.0, "undo":0.0, "clear":0.0}

    # Pinch-resize tracking
    resizing = False
    pinch_last_y = None

    save_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            index_tip = None
            detected_gesture = "none"
            handedness_label = "Right"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

                # Handedness (Left/Right)
                if result.multi_handedness:
                    handedness_label = result.multi_handedness[0].classification[0].label

                scale = palm_width(lm)
                ups = fingers_up_flags(lm, handedness_label, h)

                # Core gestures (priority order)
                if is_open_palm(ups):
                    detected_gesture = "open_palm"
                elif is_ok_sign(lm, ups, scale):
                    detected_gesture = "ok"
                elif is_thumbs_up(lm, ups, scale):
                    detected_gesture = "thumbs_up"
                elif is_thumbs_down(lm, ups, scale):
                    detected_gesture = "thumbs_down"
                elif is_pinch_resize(lm, ups, scale):
                    detected_gesture = "pinch"
                elif is_fist(ups):
                    detected_gesture = "fist"
                elif is_index_middle_ring(ups):
                    detected_gesture = "index_middle_ring"
                elif is_index_middle(ups):
                    detected_gesture = "index_middle"
                elif is_index_only(ups):
                    detected_gesture = "index_only"
                else:
                    detected_gesture = "none"

                # Smooth the gesture
                gesture_hist.append(detected_gesture)
                stable_gesture = most_common(gesture_hist)

                index_tip = lm[8]

                # === Event gestures (debounced) ===
                now = time.time()
                if stable_gesture == "ok" and now - last_event["save"] > EVENT_COOLDOWN_S:
                    last_event["save"] = now
                    composed = blend_layers(frame.copy(), drawing_layer, ALPHA_OVERLAY)
                    save_count += 1
                    filename = f"air_drawing_{save_count:03d}.png"
                    cv2.imwrite(filename, composed)
                    cv2.putText(frame, f"Saved: {filename}", (20, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
                    cv2.putText(frame, f"Saved: {filename}", (20, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                if stable_gesture in ("open_palm", "thumbs_up") and now - last_event["clear"] > EVENT_COOLDOWN_S:
                    last_event["clear"] = now
                    strokes.clear()
                    drawing_layer[:] = 0
                    current_stroke = None

                if stable_gesture == "thumbs_down" and now - last_event["undo"] > EVENT_COOLDOWN_S:
                    last_event["undo"] = now
                    if strokes:
                        strokes.pop()
                        drawing_layer = render_strokes(strokes, h, w)
                        current_stroke = None

                # === Mode gestures (continuous) ===
                if stable_gesture == "index_only":
                    if mode != "draw":
                        current_stroke = Stroke(current_color, brush_size, erase=False)
                        strokes.append(current_stroke)
                    mode = "draw"
                elif stable_gesture == "index_middle":
                    mode = "select"
                    current_stroke = None
                elif stable_gesture == "index_middle_ring":
                    if mode != "erase":
                        current_stroke = Stroke(current_color, ERASE_THICKNESS, erase=True)
                        strokes.append(current_stroke)
                    mode = "erase"
                elif stable_gesture == "fist":
                    mode = "pause"
                    current_stroke = None
                elif stable_gesture == "pinch":
                    mode = mode  # keep current tool, but enable resizing
                else:
                    # Do not force mode change on "none"
                    pass

                # === Pinch resize ===
                if stable_gesture == "pinch":
                    center_y = int((lm[4][1] + lm[8][1]) / 2)
                    if not resizing:
                        resizing = True
                        pinch_last_y = center_y
                    else:
                        dy = pinch_last_y - center_y
                        # scale change relative to palm size
                        delta = int((dy / max(8, scale)) * 60)
                        if delta != 0:
                            brush_size = int(np.clip(brush_size + delta, BRUSH_MIN, BRUSH_MAX))
                            pinch_last_y = center_y
                else:
                    resizing = False
                    pinch_last_y = None

                # === Color hover select ===
                if mode == "select" and index_tip is not None:
                    which, val = color_from_toolbar(index_tip)
                    if which == "color":
                        current_color = val
                    elif which == "eraser":
                        # quick jump to eraser
                        mode = "erase"
                        current_stroke = Stroke(current_color, ERASE_THICKNESS, erase=True)
                        strokes.append(current_stroke)

                # === Draw / Erase strokes ===
                if mode in ("draw", "erase") and index_tip is not None:
                    if current_stroke is None:
                        # create stroke if missing (e.g., after undo)
                        current_stroke = Stroke(current_color, (ERASE_THICKNESS if mode=="erase" else brush_size), erase=(mode=="erase"))
                        strokes.append(current_stroke)
                    # keep stroke tool params in sync
                    current_stroke.thickness = ERASE_THICKNESS if mode=="erase" else brush_size
                    current_stroke.erase = (mode=="erase")
                    current_stroke.color = current_color
                    current_stroke.points.append(tuple(index_tip))
                    # draw incrementally
                    if len(current_stroke.points) >= 2:
                        color = (0,0,0) if current_stroke.erase else current_stroke.color
                        cv2.line(drawing_layer, current_stroke.points[-2], current_stroke.points[-1], color, current_stroke.thickness, cv2.LINE_AA)
                else:
                    current_stroke = None

            else:
                # No hand
                gesture_hist.clear()
                current_stroke = None

            # HUD
            hud = frame.copy()
            draw_toolbar(hud, current_color, mode, brush_size)

            # Brush preview
            if index_tip is not None:
                th = ERASE_THICKNESS if mode == "erase" else brush_size
                color = (0,0,0) if mode == "erase" else current_color
                cv2.circle(hud, index_tip, max(3, th//2), (0,0,0), 3)
                cv2.circle(hud, index_tip, max(3, th//2), color, 2)
                tag = "Eraser" if mode == "erase" else "Brush"
                if resizing:
                    tag += " (Resizing)"
                cv2.putText(hud, tag, (index_tip[0]+12, index_tip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                cv2.putText(hud, tag, (index_tip[0]+12, index_tip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            # Compose and show
            out = blend_layers(hud, drawing_layer, ALPHA_OVERLAY)
            cv2.imshow("AirCanvas — Gesture Draw (q to quit)", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()