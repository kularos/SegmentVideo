# Quick Start Guide

## Setup (One-time)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test that everything works:**
   ```bash
   python test_ui.py
   ```
   
   You should see a matplotlib window with a colorful test frame and an ellipse overlay.
   Press 'A' to accept or 'S' to skip to close the window.

## Running on Your Video

### Method 1: Interactive Run Script (Easiest)

1. **Put your video in the videos folder:**
   ```bash
   mkdir videos
   cp /path/to/your/video.wmv videos/
   ```

2. **Run the script:**
   ```bash
   # Linux/Mac:
   ./run.sh
   
   # Windows:
   run.bat
   
   # Any platform:
   python run.py
   ```

3. **Follow the prompts:**
   - Choose your video from the list
   - Select pZ level (try 2 first)
   - Select interactive mode
   - Confirm and start!

### Method 2: Direct Command Line

```bash
python example_workflow.py videos/your_video.wmv 2
```

## What to Expect

1. **Loading**: Video loads with temporal downsampling (shows progress)

2. **Prediction**: Model makes initial predictions on keyframes

3. **Verification**: For each keyframe:
   - A matplotlib window appears
   - Frame is shown with model overlay (red ellipse)
   - You can:
     - Press **'A'** to accept
     - Adjust parameters in text boxes and press Enter to update
     - Press **'S'** to skip
   - Window closes and moves to next frame

4. **Interpolation**: System fills in frames between verified keyframes

5. **Complete**: Statistics and time savings displayed

## Tips

- **Start with pZ=2 or pZ=3**: This gives you a manageable number of keyframes
- **Adjust parameters by typing**: Click in text box, type new value, press Enter
- **Watch the center point**: The red dot shows the center - make sure it's on your feature
- **Use keyboard shortcuts**: 'A' is faster than clicking "Accept"

## Common Issues

**"No video files found"**
- Make sure your video is in the `./videos` directory
- Check that it has a supported extension (.mp4, .avi, .wmv, etc.)

**"Module not found"**
- Run `pip install -r requirements.txt`

**matplotlib window doesn't appear**
- Check your matplotlib backend: try setting `export MPLBACKEND=TkAgg` (Linux/Mac)
- On remote systems, you may need X11 forwarding or a display server

**Memory error**
- Use a higher pZ value (pZ=3 or pZ=4) to load fewer frames
- Check your video resolution - very high-res videos need more memory

## Next Steps

After you've verified keyframes, you can:
1. Export annotations: `state.export_annotations()`
2. Interpolate all frames: `state.interpolate_all_verified()`
3. Load interpolated data and use it for your analysis

Happy annotating! 🎬
