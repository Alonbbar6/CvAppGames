# Face Control Games

A collection of interactive games and demos controlled by facial expressions and head movements using real-time face detection.

![Face Detection Demo](Screenshot%202025-11-21%20at%209.14.53%20PM.png)

## Overview

This project demonstrates the power of computer vision and machine learning for face detection and facial landmark recognition. It includes:

- **Web-based games** controlled by facial expressions (deployed on Netlify)
- **Python applications** for face detection, blurring, and blink detection
- **Face landmark detection** using TensorFlow.js and OpenCV

## Web Games

### Available Games

| Game | Control Method | Description |
|------|----------------|-------------|
| **Flappy Bird** | Eyebrow raising | Raise your eyebrows to make the bird flap and avoid pipes |
| **Tetris** | Head movement + Eyebrows | Move head left/right to move pieces, raise eyebrows to rotate |
| **Expression Match** | Full face expressions | Match the displayed facial expression to score points |

### How It Works

1. **Camera Access** - Allow camera access when prompted. Your video stays on your device.
2. **AI Detection** - TensorFlow.js detects your face and 478 facial landmarks in real-time.
3. **Play!** - Use facial expressions and head movements to control the games.

## Project Structure

```
module14-face-detection/
├── web/                          # Web application (Netlify deployment)
│   ├── index.html               # Main landing page
│   ├── css/style.css            # Styling
│   ├── js/
│   │   ├── faceDetector.js      # TensorFlow.js face detection
│   │   ├── flappyGame.js        # Flappy Bird game logic
│   │   ├── tetrisGame.js        # Tetris game logic
│   │   └── expressionGame.js    # Expression matching game
│   └── games/                   # Individual game pages
│       ├── flappy.html
│       ├── tetris.html
│       └── expression.html
├── Applications/                 # Python Jupyter notebooks
│   └── 14_02_Face_Blurring.ipynb
├── model/                        # Pre-trained models
│   ├── haarcascade_frontalface_default.xml
│   ├── lbfmodel.yaml
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── 14_01_Face_Detection_Demo.py  # Python face detection demo
├── blink_detector.py             # Eye blink detection
├── expression_matching_game.py   # Python expression game
├── flappy_eyebrow.py            # Python Flappy Bird (eyebrow control)
├── tetris_body_control.py       # Python Tetris (body control)
└── visuals/                     # Documentation images
```

## Technologies Used

### Web Application
- **TensorFlow.js** - Machine learning in the browser
- **MediaPipe FaceMesh** - 478-point facial landmark detection
- **HTML5 Canvas** - Real-time graphics rendering
- **Vanilla JavaScript** - No framework dependencies

### Python Applications
- **OpenCV** - Computer vision operations
- **dlib** - Face detection and landmarks
- **NumPy** - Numerical computations

## Installation

### Web Application (Local Development)

```bash
# Navigate to web directory
cd web

# Start a local server (Python 3)
python -m http.server 8000

# Or use Node.js
npx serve
```

Then open http://localhost:8000 in your browser.

### Python Applications

```bash
# Install dependencies
pip install opencv-python numpy dlib

# Run face detection demo
python 14_01_Face_Detection_Demo.py

# Run blink detector
./start_blink_detector.sh
# or
python blink_detector.py

# Run expression matching game
./start_game.sh
# or
python expression_matching_game.py
```

## Deployment

The web application is configured for Netlify deployment:

```bash
# Build and deploy
netlify deploy --prod --dir=web
```

Configuration is in `web/netlify.toml`.

## Privacy

**Your privacy is protected:**
- All face detection runs locally in your browser
- No video or image data is sent to any server
- Camera feed is processed in real-time on your device

## Models

This project uses several pre-trained models:

| Model | Purpose | Source |
|-------|---------|--------|
| MediaPipe FaceMesh | 478 facial landmarks | TensorFlow.js |
| Haar Cascade | Face detection | OpenCV |
| SSD ResNet | Deep learning face detection | OpenCV DNN |
| LBF Model | Facial landmark fitting | OpenCV |

## Requirements

### Browser (Web Games)
- Modern browser with WebGL support (Chrome, Firefox, Safari, Edge)
- Webcam access

### Python
- Python 3.7+
- OpenCV 4.x
- NumPy
- dlib (optional, for landmark detection)

## License

This project is for educational purposes.

## Created By

**Alonso Bardales**

---

Built with TensorFlow.js Face Landmarks Detection