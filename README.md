# RTMPose 2D Pose Estimation on Google Colab

**Complete all-in-one package** — Includes RTMLib source + ready-to-use pose estimation pipeline.

## 🚀 Installation

### Option 1: One-Command Setup (Colab Recommended)
```bash
git clone https://github.com/pradeepj247/pj_rtmlib_2dpose.git
cd pj_rtmlib_2dpose
bash setup_colab.sh
```

### Option 2: Standard Python Package
```bash
pip install git+https://github.com/pradeepj247/pj_rtmlib_2dpose.git
```

### Option 3: Development Install
```bash
git clone https://github.com/pradeepj247/pj_rtmlib_2dpose.git
cd pj_rtmlib_2dpose
pip install -e .
```

## 📦 What's Included
✅ RTMLib source code (full library — no external download needed)
✅ Pose estimation wrapper classes (pose_estimation.py, video_processor.py)
✅ Video processing pipeline
✅ Multiple demo examples (whole body, hands, real-time, web UI)
✅ Complete setup scripts

No external dependencies needed during setup — everything is self-contained!

## 🎯 Quick Start
```python
from examples.pose_estimation import PoseEstimator

# Initialize and use
estimator = PoseEstimator(device="cuda")
result = estimator.process_image("data/demo/images/demo.jpg")
```

## 📊 Performance
✅ Detection + Pose: ~39ms per frame
✅ Real-time: ~25 FPS
✅ GPU: Tesla T4 (Colab)

## 🛠 Project Structure
```
pj_rtmlib_2dpose/
├── data/             # Demo images & videos
│   └── demo/
│       ├── images/
│       └── videos/
├── models/           # Organized model files
├── examples/         # All demo + wrapper scripts
├── rtmlib/           # Full RTMLib library
├── __init__.py       # Package marker
├── pyproject.toml    # Modern package config
├── requirements.txt  # Dependencies
├── setup.py          # Legacy installation
├── setup_colab.sh    # Automated setup
├── LICENSE           # License
└── README.md         # Documentation
```

## 📝 License
Includes RTMLib under its original license.
