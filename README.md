# RTMPose 2D Pose Estimation on Google Colab

**Complete all-in-one package** — Includes RTMLib source + ready-to-use pose estimation pipeline.

## 🚀 Installation

### Option 1: One-Command Setup (Colab Recommended)
```bash
git clone https://github.com/pradeepj247/pjpose2d.git
cd pjpose2d
bash setup_colab.sh
```

**What this does:**
- ✅ Resolves ONNX Runtime conflicts automatically
- ✅ Installs all dependencies with compatible versions
- ✅ Downloads and converts YOLOv8 model to ONNX format
- ✅ Runs validation to ensure GPU acceleration is working
- ✅ Organizes models in `models/` directory

### Option 2: Standard Python Package
```bash
pip install git+https://github.com/pradeepj247/pjpose2d.git
```

### Option 3: Development Install
```bash
git clone https://github.com/pradeepj247/pjpose2d.git
cd pjpose2d
pip install -e .
```

## 🔍 Verification
After installation, run the validation script to ensure everything is working:
```bash
python validate_installation.py
```

**Expected output:** All checks should pass with ✅, including CUDA Execution Provider availability.

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
pjpose2d/
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

<!-- Test update: Git workflow verification -->
