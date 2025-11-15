"""
Single image pose estimation demo
"""

import sys
import os

# Fix for imports - add current directory to Python path
sys.path.append(os.path.dirname(__file__))

from pose_estimation import PoseEstimator

def main():
    print("🎯 RTMPose Single Image Demo")
    print("=" * 40)
    
    # Initialize pose estimator
    estimator = PoseEstimator(device="cuda")
    
    # Get absolute path to demo image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    demo_image_path = os.path.join(project_root, "data", "demo", "images", "demo.jpg")
    
    if not os.path.exists(demo_image_path):
        print(f"❌ Demo image not found: {demo_image_path}")
        print("💡 Please run the setup script first")
        return
    
    # Process image
    result = estimator.process_image(demo_image_path, display=True)
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()

