#!/bin/bash
#
# Assisted Manual Segmentation - Run Script
# Simple wrapper around run.py for Unix-like systems
#

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are installed
python3 -c "import numpy, cv2, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages are not installed"
    echo "Please run: pip install numpy opencv-python matplotlib"
    exit 1
fi

# Run the Python script
python3 run.py "$@"
