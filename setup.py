from setuptools import setup, find_packages

setup(
    name="pjpose2d",
    version="0.1.0",
    description="2D Pose Estimation Library using RTMPose",
    author="pradeepj247",
    author_email="",  # Add your email if you want
    packages=find_packages(),
    install_requires=[
        "onnxruntime-gpu==1.19.2",
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0", 
        "numpy>=1.21.0",
        "torch>=1.8.0",
    ],
    python_requires=">=3.7",
)