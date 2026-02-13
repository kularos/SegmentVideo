@echo off
REM Assisted Manual Segmentation - Run Script
REM Simple wrapper around run.py for Windows

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3 from https://www.python.org/
    pause
    exit /b 1
)

REM Check if required Python packages are installed
python -c "import numpy, cv2, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Error: Required Python packages are not installed
    echo Please run: pip install numpy opencv-python matplotlib
    pause
    exit /b 1
)

REM Run the Python script
python run.py %*
