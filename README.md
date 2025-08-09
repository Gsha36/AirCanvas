# 🎨 AirCanvas - Draw in the Air with Hand Gestures

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://mediapipe.dev)

AirCanvas is an innovative computer vision application that transforms your webcam into a virtual drawing canvas. Using advanced hand gesture recognition, you can draw, erase, change colors, and control various drawing tools—all with simple hand movements in the air!

## ✨ Features

### 🎯 Core Drawing Capabilities
- **Air Drawing**: Draw smooth lines by pointing with your index finger
- **Dynamic Brush Sizing**: Adjust brush size using pinch gestures (3-40px range)
- **Color Palette**: Choose from 7 vibrant colors (White, Red, Green, Blue, Yellow, Purple, Black)
- **Eraser Tool**: Remove unwanted strokes with dedicated eraser mode
- **Real-time Preview**: See your brush size and mode with visual feedback

### 🖐️ Intuitive Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ **Index Only** | Draw Mode | Point with index finger to draw |
| ✌️ **Index + Middle** | Select Mode | Choose colors from toolbar |
| 🤟 **Index + Middle + Ring** | Erase Mode | Switch to eraser tool |
| ✊ **Fist** | Pause Mode | Stop all drawing activities |
| 🤏 **Pinch** | Resize Brush | Pinch thumb+index to adjust brush size |
| 👌 **OK Sign** | Save Drawing | Save current artwork as PNG |
| ✋ **Open Palm** | Clear Canvas | Erase entire drawing |
| 👍 **Thumbs Up** | Clear Canvas | Alternative clear gesture |
| 👎 **Thumbs Down** | Undo Stroke | Remove last drawn stroke |

### 🔧 Advanced Features
- **Stroke-based Undo System**: Individual stroke removal capability
- **Gesture Smoothing**: 6-frame majority vote for stable gesture detection
- **Event Cooldown**: 1.2s debouncing prevents accidental triggers
- **Layer Blending**: Transparent drawing overlay with adjustable opacity
- **Auto-numbering**: Sequential file naming for saved drawings
- **Handedness Detection**: Works with both left and right hands

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python mediapipe numpy
```

### Installation & Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gsha36/AirCanvas.git
   cd AirCanvas
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If requirements.txt doesn't exist, install manually:*
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Run the application**:
   ```bash
   python air_draw.py
   ```

4. **Start drawing**: Position your hand in front of the camera and use the gestures above!

5. **Exit**: Press 'q' or close the window to quit

## 🎮 How to Use

### Getting Started
1. **Camera Setup**: Ensure your webcam is connected and working
2. **Lighting**: Use good lighting for better hand detection
3. **Distance**: Keep your hand 1-3 feet from the camera
4. **Background**: Plain backgrounds work best for hand tracking

### Drawing Workflow
1. **Make a fist** to start in pause mode
2. **Point with index finger** to enter draw mode
3. **Move your finger** to create strokes
4. **Use two fingers** (index + middle) to enter select mode
5. **Hover over colors** in the toolbar to switch colors
6. **Make OK sign** 👌 to save your masterpiece
7. **Open palm** ✋ to clear and start fresh

### Pro Tips
- **Smooth Movements**: Slower movements create smoother lines
- **Gesture Clarity**: Make clear, distinct gestures for best recognition
- **Brush Sizing**: Use pinch gesture and move vertically to resize
- **Quick Erasing**: Three-finger gesture instantly switches to eraser

## ⚙️ Configuration

The application includes several customizable parameters in the configuration section:

```python
# Camera Settings
CAM_INDEX = 0              # Webcam index (0 for default)
FRAME_W, FRAME_H = 1280, 720  # Resolution

# Brush Settings
BRUSH_MIN, BRUSH_MAX = 3, 40   # Brush size range
DRAW_THICKNESS_INIT = 8        # Starting brush size
ERASE_THICKNESS = 40           # Eraser size

# Performance Settings
GESTURE_SMOOTH_N = 6           # Gesture smoothing frames
EVENT_COOLDOWN_S = 1.2         # Cooldown between special events
ALPHA_OVERLAY = 0.60           # Drawing transparency
```

## 🏗️ Technical Architecture

### Core Components
- **Hand Detection**: MediaPipe Hands solution for robust hand tracking
- **Gesture Recognition**: Custom finger-counting algorithm with palm-width normalization
- **Drawing Engine**: OpenCV-based stroke rendering with anti-aliasing
- **State Management**: Mode-based system with gesture history smoothing
- **File I/O**: PNG export with alpha blending

### Key Algorithms
- **Finger Detection**: Landmark-based finger up/down detection with handedness awareness
- **Gesture Smoothing**: Majority vote across multiple frames for stability
- **Pinch Detection**: Normalized distance calculation between thumb and index tips
- **Stroke Rendering**: Vector-based line drawing with thickness and color support

## 🎯 Gesture Recognition Details

The application uses a sophisticated gesture recognition system:

### Finger Detection
- **Robust Algorithm**: Uses landmark positions with handedness-aware thumb detection
- **Margin Tolerance**: 2% screen height tolerance for finger position variations
- **Palm Normalization**: All measurements scaled relative to palm width

### Gesture Hierarchy
1. **Primary Gestures**: Open palm, OK sign, thumbs up/down (highest priority)
2. **Tool Gestures**: Pinch, fist, finger combinations (medium priority)
3. **Drawing Gestures**: Index finger pointing (continuous mode)

## 🔍 Troubleshooting

### Common Issues
- **Hand not detected**: Ensure good lighting and plain background
- **Gestures not recognized**: Make clearer, more distinct hand shapes
- **Camera not working**: Check camera permissions and CAM_INDEX setting
- **Performance issues**: Reduce resolution or close other camera applications

### System Requirements
- **Python**: 3.7 or higher
- **Webcam**: Any USB or built-in camera
- **RAM**: 4GB minimum (8GB recommended)
- **OS**: Windows, macOS, or Linux

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features or gestures
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- **MediaPipe**: Google's fantastic hand tracking solution
- **OpenCV**: Computer vision library powering the graphics
- **NumPy**: Numerical computing foundation

---

**Ready to paint the air? Clone the repo and start creating! 🎨✨**