# Yolov8-Tracks
REU Project (Mac)
This guide will help you set up the necessary prerequisites for using YOLOv8. Follow the steps below to ensure you have everything in place to start working with this powerful object detection model.

Prerequisites
Before you can start using YOLOv8, you need to ensure that your system meets the following requirements and that you have installed the necessary software and libraries.

1. Python Installation
YOLOv8 is built in Python, so the first step is to have Python installed on your machine.

Python Version: YOLOv8 requires Python 3.8 or later. You can check your Python version by running:

python --version

If you do not have Python installed or need to update it, follow the steps below 
You can install Python using Homebrew. If you don't have Homebrew installed, open Terminal and run:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

After installing Homebrew, install Python with:
brew install python

2. pip Installation
pip is the package installer for Python. It should be installed automatically with Python, but you can ensure it is up-to-date by running:

python -m ensurepip --upgrade

3. Install Ultralytics YOLOv8
Ultralytics is the developer of YOLOv8. You can install the YOLOv8 library using pip.

pip install ultralytics

5. Install Additional Dependencies
YOLOv8 might require additional Python libraries depending on your use case. The core dependencies include:

torch: For tensor computations and deep learning.
opencv-python: For image and video processing.
matplotlib: For plotting.
Install these libraries using pip:

pip install torch opencv-python matplotlib

Summary
Once you have Python, pip, and the required dependencies installed, you are ready to start using YOLOv8. Here's a quick recap:

Install Python 3.8 or later.
Ensure pip is installed and up-to-date.
Install Ultralytics YOLOv8 using pip install ultralytics.
Install additional dependencies: torch, opencv-python, and matplotlib.
You can now proceed to use YOLOv8 for your object detection tasks. For detailed instructions and tutorials on using YOLOv8, refer to the official Ultralytics documentation.
