# Assisted Manual Segmentation for Video Feature Tracking

A Python framework for easing manual video annotation by leveraging temporal correlation and intelligent interpolation.

## Core Concept

Instead of manually tracking features in every frame, this system:

1. **Temporally downsamples** video to a "correlation time" scale using `pZ` parameter
2. **Model predicts** primitive shapes (ellipsoids, curves, etc.) on keyframes
3. **User verifies** and corrects predictions (much faster than manual tracking)
4. **Temporal consistency**: Each frame uses the last accepted position as starting point
5. **Interpolates** between verified keyframes to fill all frames

**Temporal Consistency Advantage:**
- Each new frame starts from your last accepted position
- Greatly reduces the amount of adjustment needed per frame
- Especially powerful for smooth motion (worm locomotion, cell tracking, etc.)
- Can cut annotation time per frame by ~50%

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
run_chain.py            - Chain/curve tracking script with N segments
run.sh                  - Bash wrapper for Unix/Linux/Mac
run.bat                 - Batch wrapper for Windows
video_loader.py         - Load video into (3, W, H, F) tensor with pZ downsampling
annotation_state.py     - Manage annotations, verification, interpolation, and interactive UI with drag support
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
from video_loader import load_video_to_tensor

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
from annotation_state import AnnotationState, create_ellipsoid_params
from video_loader import get_video_info

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
from annotation_state import verify_with_ui

# Show frame with prediction overlay
accepted, corrected_params = verify_with_ui(
    frame,              # (3, W, H) tensor
    predicted_params,   # ModelParameters with prediction
    frame_idx=0,        # Original frame index
    tensor_idx=0,       # Index in downsampled tensor
    total_frames=10     # Total keyframes (for progress display)
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

## Tracking Modes

### 1. Ellipsoid Tracking (default)

Track 3D ellipsoids (displayed as 2D ellipse projections):

```bash
python example_workflow.py video.wmv 2
```

**Drag controls:**
- Center point (red dot): Move the ellipsoid
- Axis point (red square): Rotate and resize

### 2. Chain/Curve Tracking (NEW!)

Track linear features, filaments, or curves with N evenly-spaced segments:

```bash
# Track with 5 segments (6 control points)
python run_chain.py video.wmv 5 2

# Track with 10 segments (11 control points)
python run_chain.py video.wmv 10 2
```

**Drag controls with constant spacing constraint:**
- **Endpoints (red squares)**: Drag freely to adjust chain length and orientation
- **Middle points (red circles)**: Drag perpendicular to the backbone only
- Spacing between points remains constant (evenly distributed)
- Dashed line shows the straight backbone between endpoints

**Intelligent rotation using complex numbers:**
When you drag an endpoint, the chain's curvature shape is preserved:
- Perpendicular offsets rotate with the chain orientation
- A chain bent to the left stays bent left even when rotated
- Uses complex number multiplication: `offset_new = offset_old × e^(iθ)`
- Natural, intuitive behavior that maintains the chain's "shape"

**Why constant spacing?**
This constraint ensures:
- Points stay evenly distributed along the chain
- Natural representation of physical chains/filaments
- More stable tracking and interpolation
- Easier to model bending while maintaining chain length

**Use cases:**
- C. elegans worm tracking (preserves body curvature during turns)
- Fiber/filament tracking in microscopy
- Blood vessel segmentation  
- Root/neurite tracing
- Any linear or chain-like structure

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

### Recommended approach with temporal consistency:

1. **Start coarse** (pZ=3 or pZ=4): Get global motion, identify difficult regions
2. **First frame**: Position the model on your feature
3. **Subsequent frames**: Model automatically starts from your last accepted position
4. **Minor adjustments**: Most frames need only small corrections (if any)
5. **Refine locally** (pZ=2 or pZ=1): Add intermediate keyframes where motion is complex
6. **Dense interpolation** (pZ=0): Only for regions where interpolation fails

**Temporal consistency in action:**
```
Frame 0:   [Initial placement - takes 10s]
Frame 4:   [Model uses Frame 0 position - adjust takes 3s]
Frame 8:   [Model uses Frame 4 position - adjust takes 2s]
Frame 12:  [Model uses Frame 8 position - adjust takes 2s]
...
```

### Time savings example:

For 3600-frame video at 30fps (2 minutes):

| Approach | Frames to annotate | Time @ 30s/frame | Time w/ verify | Time w/ temporal |
|----------|-------------------|------------------|----------------|------------------|
| Pure manual (pZ=0) | 3600 | 30 hours | - | - |
| pZ=2 (every 4th) | 900 | - | 2.5 hours | 1.25 hours |
| pZ=3 (every 8th) | 450 | - | 1.25 hours | 0.625 hours |

**Temporal consistency benefit:** Reduces per-frame adjustment time by ~50%
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

### Chain Rotation Mathematics

The chain tracking uses complex numbers to intelligently preserve curvature when endpoints are rotated:

**Problem:** When you drag an endpoint, how should the middle points move?

**Solution:** Rotate the perpendicular offsets by the chain's rotation angle.

**Math:**
```
1. Represent vectors as complex numbers: z = x + iy
2. Old chain direction: z_old = (end - start)_old
3. New chain direction: z_new = (end - start)_new  
4. Rotation: θ = arg(z_new / z_old)
5. Rotate each offset: offset_new = offset_old × e^(iθ)
```

**Result:** A bent chain stays bent in the same way, just rotated. Try `python demo_complex_rotation.py` to see a visualization!

## Interactive UI Controls

When the verification window appears:

**Keyboard Shortcuts:**
- `A` - Accept the prediction (with any modifications)
- `S` - Skip this frame

**Mouse Controls (NEW!):**
- **Left-click and drag** control points to adjust model:
  - **Ellipsoid**: Drag red center dot to move, drag red square to rotate/resize
  - **Curve/Chain (constant spacing)**: 
    - Drag endpoints (squares) freely to adjust length/orientation
    - Drag middle points (circles) perpendicular to backbone only
- Real-time visual feedback as you drag

**Parameter Adjustment (for ellipsoid models):**
- **Center X / Y**: Adjust the center position of the ellipse
- **Width / Height**: Adjust the semi-axis lengths
- Type new values in text boxes and press Enter to update
- Or just drag the control points with your mouse!

**Visual Elements:**
- Red ellipse/curve: Current prediction/adjustment
- Red dots: Draggable control points
- Red square (ellipsoid): Axis endpoint for rotation/scaling
- Red line (ellipsoid): Major axis direction
- Progress indicator: Shows which keyframe you're on (X of Y)

## Future Enhancements

- [x] Interactive GUI for user verification (matplotlib-based)
- [x] Mouse-based dragging for control points
- [x] Chain/curve tracking with N segments
- [ ] Spline interpolation for smoother motion
- [ ] Automatic keyframe selection based on optical flow
- [ ] Multi-object tracking support
- [ ] Export to common annotation formats (COCO, YOLO, etc.)
- [ ] Undo/redo for user corrections
- [ ] Batch processing multiple videos
- [ ] Support for more primitive models (cylinders, planes, polygons)
- [ ] Bezier curve support for smoother chains
