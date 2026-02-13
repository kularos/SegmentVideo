# Assisted Manual Segmentation for Video Feature Tracking

A Python framework for easing manual video annotation by leveraging temporal correlation and intelligent interpolation.

## Core Concept

Instead of manually tracking features in every frame, this system:

1. **Temporally downsamples** video to a "correlation time" scale using `pZ` parameter
2. **Model predicts** primitive shapes (ellipsoids, curves, etc.) on keyframes
3. **User verifies** and corrects predictions (much faster than manual tracking)
4. **Interpolates** between verified keyframes to fill all frames

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install numpy opencv-python matplotlib
   ```

3. **Test the installation:**
   ```bash
   python test_ui.py
   ```
   
   If a matplotlib window appears with a test frame, you're ready to go!

## Key Parameter: pZ (Temporal Zoom)

The `pZ` parameter controls temporal downsampling using base-2 logarithm:

- `pZ = 0`: Load every frame (no downsampling)
- `pZ = 1`: Load every 2nd frame (2^1 = 2)
- `pZ = 2`: Load every 4th frame (2^4 = 4)
- `pZ = 3`: Load every 8th frame (2^8 = 8)

Choose `pZ` based on your feature's "correlation time" - the timescale below which changes are negligible.

### Memory Impact

For a 720p video (1280×720) at 30fps:

| Duration | pZ=0 (all frames) | pZ=2 (every 4th) | pZ=3 (every 8th) |
|----------|-------------------|------------------|------------------|
| 60s      | ~17.8 GB         | ~4.5 GB         | ~2.2 GB         |
| 120s     | ~35.5 GB         | ~8.9 GB         | ~4.4 GB         |

## File Structure

```
run.py                  - Interactive run script (recommended)
run.sh                  - Bash wrapper for Unix/Linux/Mac
run.bat                 - Batch wrapper for Windows
video_loader.py         - Load video into (3, W, H, F) tensor with pZ downsampling
annotation_state.py     - Manage annotations, verification, interpolation, and interactive UI
example_workflow.py     - Complete pipeline demonstration with interactive verification
test_ui.py             - Simple test to try the interactive UI without a video file
```

## Quick Start

### Easy Way: Use the Run Script

1. **Place your videos** in the `./videos` directory
2. **Run the script**:
   ```bash
   # On Linux/Mac:
   ./run.sh
   
   # On Windows:
   run.bat
   
   # Or use Python directly (all platforms):
   python run.py
   ```

3. The script will:
   - Show you available videos to choose from
   - Let you select the pZ level (temporal zoom)
   - Let you choose interactive UI or auto-accept mode
   - Run the complete workflow

### Test the Interactive UI (No Video Required)

```bash
python test_ui.py
```

This creates a synthetic test frame and opens the verification window so you can try out the interface before working with real video.

## Usage

### Easy Method: Run Script

The simplest way to use the system:

1. **Create videos directory and add your video files:**
   ```bash
   mkdir videos
   cp /path/to/your/video.wmv videos/
   ```

2. **Run the interactive script:**
   ```bash
   # On Linux/Mac:
   ./run.sh
   
   # On Windows:
   run.bat
   
   # Or directly with Python (all platforms):
   python run.py
   ```

3. **Follow the prompts:**
   - Select which video to process
   - Choose pZ level (2 is recommended for most cases)
   - Choose interactive UI or auto-accept mode
   - The workflow will run automatically

**Supported video formats:** .mp4, .avi, .mov, .wmv, .mkv, .flv, .webm, .m4v, .mpg, .mpeg

### Manual Method: Direct Python Usage

### 1. Basic Video Loading

```python
from lib.video import load_video_to_tensor

# Load every 4th frame (pZ=2)
tensor, frame_indices = load_video_to_tensor("video.wmv", pZ=2)

# tensor shape: (3, W, H, F)
#   3: RGB channels (0-1 float)
#   W: Width (left to right)
#   H: Height (bottom to top)
#   F: Number of frames (downsampled)

# frame_indices maps tensor index to original frame number
original_frame = frame_indices[10]  # Original frame number for tensor[:,:,:,10]
```

### 2. Annotation State Management

```python
from lib.annotation import AnnotationState, create_ellipsoid_params
from lib.video import get_video_info

# Initialize state
info = get_video_info("video.wmv")
state = AnnotationState(info['frames'], frame_indices)

# Store model prediction
params = create_ellipsoid_params(
   frame_idx=0,
   center=[100, 200, 50],
   semi_axes=[20, 15, 10],
   rotation=[0, 0, 0]
)
state.set_prediction(0, params)

# User verifies (with optional corrections)
state.verify_annotation(0, corrected_params)

# Interpolate between verified keyframes
state.interpolate_all_verified()

# Check progress
progress = state.get_progress()
print(f"Annotated {progress['n_annotated']}/{progress['total_frames']} frames")
```

### 3. Interactive Verification UI

The system includes a matplotlib-based interactive UI for verifying predictions:

```python
from lib.annotation import verify_with_ui

# Show frame with prediction overlay
accepted, corrected_params = verify_with_ui(
   frame,  # (3, W, H) tensor
   predicted_params,  # ModelParameters with prediction
   frame_idx=0,  # Original frame index
   tensor_idx=0,  # Index in downsampled tensor
   total_frames=10  # Total keyframes (for progress display)
)

if accepted:
   state.verify_annotation(frame_idx, corrected_params)
```

**UI Features:**
- Displays frame with model overlay (ellipse for ellipsoid models)
- Keyboard shortcuts: `A` to accept, `S` to skip
- Interactive parameter adjustment via text boxes
- Real-time preview of parameter changes
- Shows progress (keyframe X of Y)

### 4. Complete Workflow

```bash
# Run example workflow with interactive UI
python example_workflow.py video.wmv 2

# Or disable UI for testing (auto-accepts all predictions)
python example_workflow.py video.wmv 2 --no-ui
```

This demonstrates:
- Loading video with temporal downsampling
- Running predictions on keyframes
- **Interactive user verification with matplotlib UI**
- Interpolation to all frames
- Time savings calculation

## Primitive Models Supported

### Ellipsoid
```python
create_ellipsoid_params(
    frame_idx=0,
    center=[x, y, z],           # Center position
    semi_axes=[a, b, c],        # Semi-axis lengths
    rotation=[roll, pitch, yaw] # Rotation angles (radians)
)
```

### Curve (Evenly-spaced points)
```python
create_curve_params(
    frame_idx=0,
    points=[[x1,y1,z1], [x2,y2,z2], ...]  # N points defining curve
)
```

### Custom Models
The `ModelParameters` class is flexible - you can define any model by storing arbitrary parameters in the `parameters` dict:

```python
ModelParameters(
    frame_idx=0,
    model_type="custom_shape",
    parameters={
        'param1': np.array([...]),
        'param2': np.array([...]),
    }
)
```

## Coordinate System

- **X (Width)**: Left to right, increasing
- **Y (Height)**: **Bottom to top**, increasing (note: flipped from standard image coordinates)
- **Z (Depth)**: Into the scene (for 3D models)
- **T (Time)**: First frame to last frame, increasing

## Interpolation Strategy

Linear interpolation between verified keyframes:

```
Frame:     0    4    8    12   16
Status:    [V]  [ ]  [V]  [ ]  [V]
           ↓    ↓    ↓    ↓    ↓
After:     [V]  [I]  [V]  [I]  [V]

V = Verified by user
I = Interpolated
```

For frame `f` between verified frames `f1` and `f2`:
```
alpha = (f - f1) / (f2 - f1)
param(f) = (1 - alpha) * param(f1) + alpha * param(f2)
```

## Workflow Strategy

### Recommended approach:

1. **Start coarse** (pZ=3 or pZ=4): Get global motion, identify difficult regions
2. **Refine locally** (pZ=2 or pZ=1): Add intermediate keyframes where motion is complex
3. **Dense interpolation** (pZ=0): Only for regions where interpolation fails

### Time savings example:

For 3600-frame video at 30fps (2 minutes):

| Approach | Frames to annotate | Time @ 30s/frame | Time @ 5s/verify |
|----------|-------------------|------------------|------------------|
| Pure manual (pZ=0) | 3600 | 30 hours | - |
| pZ=2 (every 4th) | 900 | - | 1.25 hours |
| pZ=3 (every 8th) | 450 | - | 0.625 hours |

**Savings: 96-98% reduction in annotation time**

## Implementation Details

### Tensor Format: (3, W, H, F)

This format is chosen for efficient access patterns:
- **Channel-first**: Natural for RGB operations
- **Width before Height**: Matches typical graphics conventions
- **Frame last**: Allows efficient temporal slicing `tensor[:,:,:,t1:t2]`

### Memory Efficiency

- **float32**: 4 bytes per value
- **Total memory**: `3 × W × H × F × 4` bytes
- **Example** (720p, 60s @ 30fps, pZ=2): 
  - `3 × 1280 × 720 × 450 × 4 = ~4.5 GB`

### Extensibility

The system is designed to be extended with:
- Custom primitive models
- Non-linear interpolation schemes (spline, physics-based)
- Multi-object tracking
- Automatic keyframe selection based on motion detection
- Active learning to suggest which keyframes need verification

## Interactive UI Controls

When the verification window appears:

**Keyboard Shortcuts:**
- `A` - Accept the prediction (with any modifications)
- `S` - Skip this frame

**Parameter Adjustment (for ellipsoid models):**
- **Center X / Y**: Adjust the center position of the ellipse
- **Width / Height**: Adjust the semi-axis lengths
- Type new values in text boxes and press Enter to update
- The overlay updates in real-time

**Visual Elements:**
- Red ellipse: Current prediction/adjustment
- Red dot: Center point
- Red line: Major axis direction
- Progress indicator: Shows which keyframe you're on (X of Y)

## Future Enhancements

- [x] Interactive GUI for user verification (matplotlib-based)
- [ ] Enhanced GUI with mouse-based dragging/resizing
- [ ] Spline interpolation for smoother motion
- [ ] Automatic keyframe selection based on optical flow
- [ ] Multi-object tracking support
- [ ] Export to common annotation formats (COCO, YOLO, etc.)
- [ ] Undo/redo for user corrections
- [ ] Batch processing multiple videos
- [ ] Support for more primitive models (cylinders, planes, polygons)
