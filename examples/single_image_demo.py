
"""
Single image pose estimation demo
"""

import sys
import os
sys.path.append('..')

from pose_estimation import PoseEstimator

def main():
    print("🎯 RTMPose Single Image Demo")
    print("=" * 40)
    
    # Initialize pose estimator
    estimator = PoseEstimator(device="cuda")
    
    # Process demo image
    demo_image_path = "rtmlib/demo.jpg"
    
    if not os.path.exists(demo_image_path):
        print(f"❌ Demo image not found: {demo_image_path}")
        print("💡 Make sure you're in the rtmlib directory")
        return
    
    # Process image
    result = estimator.process_image(demo_image_path, display=True)
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
