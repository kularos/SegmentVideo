# Migration Guide

This guide helps you migrate from the old codebase to the refactored architecture.

## File Mappings

### Old → New

| Old File | New Location | Notes |
|----------|--------------|-------|
| `video.py` | `segmentvideo/io/video_loader.py` | No API changes |
| `cv.py` | `segmentvideo/utils/watershed.py` + `segmentvideo/workflows/segmentation.py` | Refactored into reusable components |
| `annotation.py` | `segmentvideo/annotation/state.py` + `segmentvideo/models/curve.py` | Separated model from state |
| `run.py` | `main.py segment` | Unified CLI |
| `run_chain.py` | `main.py track` | Not yet implemented |

## API Changes

### Video Loading

**Old:**
```python
from video import load_video_to_tensor

tensor, indices = load_video_to_tensor("video.wmv", pZ=2)
```

**New:**
```python
from segmentvideo.io import load_video_to_tensor

tensor, indices = load_video_to_tensor("video.wmv", pZ=2)
```

### Watershed Segmentation

**Old:**
```python
# cv.py had everything in one file with WatershedManager class
manager = WatershedManager(ax)
# ... interactive UI code mixed with logic
```

**New:**
```python
from segmentvideo.utils import WatershedSegmenter
from segmentvideo.workflows import run_segmentation_workflow

# Option 1: Programmatic use
segmenter = WatershedSegmenter(image)
segmenter.add_seed(x, y, marker_id)
markers = segmenter.run_watershed()
contour = segmenter.get_feature_edge_contour(marker_id, edge_point)

# Option 2: Interactive workflow
curve_model = run_segmentation_workflow(image, n_curve_points=10)
```

### Curve Model

**Old:**
```python
# Probably had curve parameters as dict or numpy arrays
curve_params = {
    'points': np.array([[x1, y1], [x2, y2], ...])
}
```

**New:**
```python
from segmentvideo.models import CurveModel

# Create from points
curve = CurveModel(frame_idx=0, points=np.array([[x1, y1], [x2, y2], ...]))

# Create from contour with resampling
curve = CurveModel.from_contour(frame_idx=0, contour_points=contour, n_points=10)

# Interpolate
interpolated = curve1.interpolate(curve2, alpha=0.5)

# Render
curve.render(ax, color='red', linewidth=2)

# Get control points for dragging
control_points = curve.get_control_points()

# Update from drag
curve.update_from_control_point(point_index=0, new_x=100, new_y=200)
```

### Annotation State

**Old:**
```python
# Probably had annotation tracking in annotation.py
# Implementation details unknown
```

**New:**
```python
from segmentvideo.annotation import AnnotationState

# Initialize
state = AnnotationState(total_frames=3600, frame_indices=frame_indices)

# Set prediction
state.set_prediction(frame_idx=0, model=curve_model)

# Verify
state.verify_annotation(frame_idx=0, corrected_model=corrected_curve)

# Get last verified for temporal consistency
last_model = state.get_last_verified_model(before_frame=10)

# Interpolate between keyframes
state.interpolate_all_verified()

# Save/load
state.save_to_file("annotations.json")
state = AnnotationState.load_from_file("annotations.json")

# Progress
state.print_progress()
stats = state.get_progress()
```

## Command Line Interface

### Old

```bash
# Multiple separate scripts
python run.py
python run_chain.py video.wmv 10 2
python example_workflow.py video.wmv 2
```

### New

```bash
# Unified CLI
python main.py segment video.wmv --pz 2 --n-points 10
python main.py track video.wmv annotations.json --pz 2
python main.py test-ui

# Or after installation
segmentvideo segment video.wmv --pz 2 --n-points 10
```

## Installation

### Old

```bash
# Just run scripts directly
python run.py
```

### New

```bash
# Proper package installation
pip install -e .

# Then use from anywhere
segmentvideo segment video.wmv
```

## Key Improvements

1. **Separation of Concerns**
   - Models (`CurveModel`) separate from state (`AnnotationState`)
   - Watershed logic (`WatershedSegmenter`) separate from UI
   - Video loading isolated in `io/` package

2. **Extensibility**
   - Easy to add new model types (inherit from `BaseModel`)
   - Plugin architecture for interpolation strategies
   - Clean interfaces between components

3. **Testability**
   - Each component can be tested independently
   - Mock-friendly design
   - Unit tests can be added easily

4. **Reusability**
   - Import and use components programmatically
   - Not tied to specific UI framework
   - Can integrate into larger pipelines

5. **Documentation**
   - Type hints throughout
   - Docstrings for all public methods
   - Clear API contracts

## Migration Steps

1. **Update imports:**
   ```python
   # Old
   from video import load_video_to_tensor
   
   # New
   from segmentvideo.io import load_video_to_tensor
   ```

2. **Refactor watershed code:**
   - Move logic from `cv.py` to use `WatershedSegmenter`
   - Use `run_segmentation_workflow` for interactive use

3. **Update curve handling:**
   - Replace dict/array representations with `CurveModel` objects
   - Use `CurveModel.from_contour()` to create from watershed output

4. **Update annotation tracking:**
   - Replace custom tracking with `AnnotationState`
   - Use status enums instead of boolean flags

5. **Update CLI:**
   - Replace old run scripts with `main.py` commands
   - Update documentation to reference new CLI

## Backwards Compatibility

If you need to maintain compatibility with old code:

```python
# Create wrapper functions
def old_style_watershed(image):
    """Wrapper for old cv.py interface."""
    from segmentvideo.utils import WatershedSegmenter
    segmenter = WatershedSegmenter(image)
    # ... wrap new API to match old behavior
    return segmenter

# Or use both
import sys
sys.path.insert(0, 'path/to/old/code')
from old_code import something
from segmentvideo.models import CurveModel  # Use new code too
```

## Questions?

If you encounter issues during migration:

1. Check the examples in `examples/`
2. Read the docstrings in the source code
3. Open an issue on GitHub
