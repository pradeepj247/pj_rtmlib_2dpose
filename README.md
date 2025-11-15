# RTMPose 2D Pose Estimation on Google Colab

Real-time 2D human pose estimation using RTMPose-M and YOLOv8 with GPU acceleration.

## 🚀 Quick Start (Google Colab)

### 1. One-Click Setup
```python
!git clone https://github.com/YOUR_USERNAME/my-rtmpose-project.git
%cd my-rtmpose-project
!bash setup_colab.sh

### 2. Run Single Image Demo
from pose_estimation import PoseEstimator
estimator = PoseEstimator()
estimator.process_image("path/to/image.jpg")

### 3. Run Video Processing
from video_processor import VideoProcessor
processor = VideoProcessor()
processor.process_video("input.mp4", "output.mp4")

📊 Performance

Detection + Pose: ~39ms per frame

Real-time: ~25 FPS

GPU: Tesla T4 (Colab)

🛠 Features
Single image pose estimation

Video processing with pose tracking

Real-time performance

Multiple person detection

GPU acceleration

📁 Project Structure
text
my-rtmpose-project/
├── setup_colab.sh          # One-click setup script
├── requirements.txt        # Python dependencies
├── pose_estimation.py     # Core pose estimation
├── video_processor.py     # Video processing
└── examples/
    ├── single_image_demo.py
    └── video_demo.py
🔧 Manual Setup
See setup_colab.sh for detailed installation steps.

📝 License
MIT License
