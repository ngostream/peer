# Peer - Visual Accountability

A real-time focus monitoring system that tracks your posture and attention during work sessions.

## Features

- Real-time focus score tracking
- Posture-based distraction detection
- Clean, modern dashboard UI
- Instant demo mode (no setup required)

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ngostream/peer
cd peer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

4. Open your browser:
```
http://localhost:8000
```

5. Click "LOGIN WITH GOOGLE" to instantly log in (mock authentication for demo)

## Usage

1. **Login**: Click the login button to access the dashboard (instant mock login)
2. **Start Session**: Click "Start Session" to begin monitoring
3. **Monitor**: Watch your focus score update in real-time

## Demo Mode

The application runs in demo mode with:
- Simulated camera feed (no camera required)
- Mock authentication (one-click login)
- Real-time focus score simulation

Perfect for hackathon demonstrations where you need a working demo without complex setup.

## Project Structure

```
peer/
├── main.py              # FastAPI application
├── camera.py            # Video camera and focus detection
├── templates/
│   └── index.html      # Frontend dashboard
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: HTML, TailwindCSS, JavaScript
- **Computer Vision**: OpenCV

## Future Enhancements

- Google OAuth authentication
- Friend connections and social features
- Leaderboards and comparisons
- Mobile app support

## License

Built for SB Hacks XII
