# Webcam Hands Zooming - Teams Gesture Camera

A virtual camera application that enables hand gesture-based zoom control for video calls and streaming. Perfect for use with OBS Studio, Microsoft Teams, Zoom, and other video applications.

## 🚀 Features

- **Hand Gesture Control**: Use natural hand movements to control zoom
  - Close hand (thumb + index finger together) = Zoom OUT
  - Open hand (spread thumb and index finger apart) = Zoom IN
- **Multi-Hand Support**: Works with multiple hands for enhanced control
- **Virtual Camera Output**: Compatible with OBS Studio and video conferencing apps
- **Real-time Display**: Shows zoom level, hand detection, and FPS
- **Sound Effects**: Audio feedback during zoom transitions
- **Resolution Independent**: Automatically adapts to your camera's resolution
- **Smooth Zoom**: Advanced smoothing algorithms for stable zoom experience

## 📋 Requirements

- **Python 3.10+** (tested with Python 3.11, 3.12)
- **Webcam** (any USB or built-in camera)
- **Windows** (tested on Windows 10/11)

## 🛠️ Installation

### Method 1: Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lordsib/webcam_hands_zooming.git
   cd webcam_hands_zooming
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Install dependencies**:
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python teamsGestureCamera.py
   ```

## 🎮 Usage

### Basic Controls

- **Close your hand** (bring thumb and index finger together) → **Zoom OUT**
- **Open your hand** (spread thumb and index finger apart) → **Zoom IN**
- Multiple hands work together for enhanced control

### Keyboard Shortcuts

- **`q`** - Quit the application
- **`r`** - Reset zoom to 1.0x
- **`i`** - Toggle interface display on/off
- **`x`** - Toggle gesture functionality on/off (keeps camera running)

### Using with OBS Studio

1. **Start the gesture camera** by running the application
2. **In OBS Studio**:
   - Add a "Window Capture" source
   - Select "Teams Gesture Camera - Virtual Output"
   - Your gesture-controlled camera is now available in OBS!

### Using with Video Conferencing Apps

1. **Start the gesture camera application**
2. **In your video app** (Teams, Zoom, etc.):
   - Go to camera settings
   - Select "OBS Virtual Camera" or look for the gesture camera window
   - Enjoy gesture-controlled zoom during your calls!

## 📦 Dependencies

The application requires these Python packages (automatically installed with `requirements.txt`):

- **opencv-python** - Computer vision and camera operations
- **mediapipe** - Hand detection and tracking
- **numpy** - Numerical computations
- **pygame** - Sound effects playback

## 🎯 How It Works

1. **Camera Input**: Captures video from your webcam
2. **Hand Detection**: Uses MediaPipe to detect and track hand landmarks
3. **Gesture Recognition**: Calculates distance between thumb and index finger
4. **Zoom Control**: Maps hand gestures to zoom levels (0.5x to 3.0x)
5. **Virtual Output**: Displays the zoomed video in a window that can be captured by OBS

## 🔧 Configuration

### Gesture Sensitivity

The application automatically adjusts gesture sensitivity based on your camera resolution:
- **Full HD (1920x1080)**: 100-500 pixel range
- **Other resolutions**: Automatically scaled

### Sound Effects

- Place your sound file as `pam-pam-pammm.mp3` in the main directory
- Sound plays when smoothly zooming from 1x to 2.5x
- Audio is routed to speakers (can be configured for virtual microphone)

## 🐛 Troubleshooting

### Camera Not Detected
- Make sure your webcam is connected and not used by another application
- Try closing other video applications (Skype, Teams, etc.)
- Check Windows camera privacy settings

### Hand Detection Issues
- Ensure good lighting conditions
- Keep hands clearly visible to the camera
- Avoid complex backgrounds behind your hands

### Performance Issues
- The application automatically optimizes for your camera's resolution
- Higher resolutions may require more processing power
- Check the FPS display in the bottom-right corner

### Virtual Environment Issues
If you encounter Python package errors:
```bash
# Delete the existing venv folder and recreate
rmdir /s venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

```
webcam_hands_zooming/
├── teamsGestureCamera.py      # Main application
├── requirements.txt           # Python dependencies
├── run_gesture_camera.bat     # Easy launcher script
├── pam-pam-pammm.mp3         # Sound effect file
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source. Feel free to use, modify, and distribute as needed.

---

**Enjoy gesture-controlled video calls!** 🎥✋📹
