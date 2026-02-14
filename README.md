# SegmentVideo - Assisted Manual Video Segmentation

A Python framework for easing manual video annotation by leveraging temporal correlation, intelligent interpolation, and watershed-based initial segmentation.

## Key Features

- **Watershed Segmentation**: Interactive seed-based segmentation on first frame
- **Automatic Curve Fitting**: Extract edge contours and fit curve models
- **Temporal Consistency**: Each frame starts from last verified position
- **Smart Interpolation**: Fill frames between verified keyframes
- **Flexible Architecture**: Extensible model system for different primitives

## Installation

### From Source

```bash
git clone https://github.com/kularos/SegmentVideo.git
cd SegmentVideo
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.3.0

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Segment First Frame

Run interactive watershed segmentation on the first frame to create initial curve model:

```bash
python main.py segment path/to/video.wmv --pz 2 --n-points 10
```

This will:
1. Load the first frame
2. Open interactive UI for seed placement
3. Run watershed segmentation
4. Allow you to select the edge you want to track
5. Fit a curve model to that edge
6. Save the initial curve model

### 2. Track Through Frames (Coming Soon)

```bash
python main.py track path/to/video.wmv initial_curve.json --pz 2
```

### 3. Test UI

Test the interface without a video file:

```bash
python main.py test-ui
```

## Workflow

### Step 1: Watershed Segmentation

Place seeds on the first frame to identify features:

- **Click**: Add new seed
- **Drag**: Move seed
- **Click on seed**: Remove seed
- **Radio buttons**: Select Background / Feature 1 / Feature 2
- **Update Mask**: Run watershed segmentation

### Step 2: Edge Selection

After watershed completes, click on the edge of Feature 1 that you want to track.

### Step 3: Curve Fitting

Click "Fit Curve" to extract the edge contour and fit a curve model with N evenly-spaced control points.

### Step 4: Tracking (Coming Soon)

Track the curve model through subsequent frames with interactive verification.

## Project Structure

```
SegmentVideo/
├── segmentvideo/              # Main package
│   ├── __init__.py
│   ├── io/                    # Video loading
│   │   ├── __init__.py
│   │   └── video_loader.py
│   ├── models/                # Tracking primitives
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base model
│   │   └── curve.py          # Curve/chain model
│   ├── annotation/            # State management
│   │   ├── __init__.py
│   │   └── state.py
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   └── watershed.py      # Watershed segmentation
│   └── workflows/             # Integrated workflows
│       ├── __init__.py
│       └── segmentation.py   # Watershed→Curve pipeline
├── tests/                     # Unit tests
├── examples/                  # Example workflows
├── docs/                      # Documentation
├── main.py                    # Unified CLI entry point
├── setup.py                   # Package installation
├── pyproject.toml            # Modern Python packaging
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Architecture

### Models

All tracking models inherit from `BaseModel`:

```python
from segmentvideo.models import CurveModel

# Create from contour
curve = CurveModel.from_contour(
    frame_idx=0,
    contour_points=contour,
    n_points=10
)

# Interpolate between two curves
interpolated = curve1.interpolate(curve2, alpha=0.5)

# Render on matplotlib axes
curve.render(ax, color='red', linewidth=2)
```

### Watershed Segmentation

```python
from segmentvideo.utils import WatershedSegmenter

# Initialize
segmenter = WatershedSegmenter(image)

# Add seeds
segmenter.add_seed(x=100, y=200, marker_id=2)  # Feature 1

# Run watershed
markers = segmenter.run_watershed()

# Extract edge contour
contour = segmenter.get_feature_edge_contour(
    marker_id=2, 
    edge_point=(x, y)
)
```

### Annotation State

```python
from segmentvideo.annotation import AnnotationState

# Initialize
state = AnnotationState(total_frames=3600, frame_indices=frame_indices)

# Store prediction
state.set_prediction(frame_idx, model)

# Verify with corrections
state.verify_annotation(frame_idx, corrected_model)

# Interpolate between keyframes
state.interpolate_all_verified()

# Save/load
state.save_to_file("annotations.json")
state = AnnotationState.load_from_file("annotations.json")
```

## Temporal Zoom (pZ Parameter)

The `pZ` parameter controls temporal downsampling:

- `pZ = 0`: Every frame (no downsampling)
- `pZ = 1`: Every 2nd frame (2^1 = 2)
- `pZ = 2`: Every 4th frame (2^2 = 4)  ← **Recommended**
- `pZ = 3`: Every 8th frame (2^3 = 8)

Choose based on your feature's correlation time.

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

### Code Formatting

```bash
black segmentvideo/
isort segmentvideo/
flake8 segmentvideo/
```

### Type Checking

```bash
mypy segmentvideo/
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this software in your research, please cite:

```
[Add citation here]
```

## Contact

- GitHub: [@kularos](https://github.com/kularos)
- Issues: [GitHub Issues](https://github.com/kularos/SegmentVideo/issues)
