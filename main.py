#!/usr/bin/env python3
"""
Unified CLI entry point for SegmentVideo.

Usage:
    python main.py segment <video_file> [--pz PZ] [--n-points N]
    python main.py track <video_file> <annotation_file> [--pz PZ]
    python main.py test-ui
"""

import argparse
import sys
from pathlib import Path

def cmd_segment(args):
    """Run segmentation workflow on first frame."""
    from segmentvideo.io import load_video_to_tensor
    from segmentvideo.workflows import run_segmentation_workflow
    
    print(f"Loading video: {args.video}")
    print(f"  pZ (temporal zoom): {args.pz}")
    print(f"  Number of curve points: {args.n_points}")
    print(f"  Box fitting method: {args.box_method}")
    
    # Load just the first frame for segmentation
    tensor, frame_indices = load_video_to_tensor(args.video, pZ=args.pz)
    
    # Extract first frame (3, W, H) -> (H, W, 3)
    first_frame_tensor = tensor[:, :, :, 0]
    first_frame = first_frame_tensor.transpose(2, 1, 0)  # (3, W, H) -> (H, W, 3)
    
    print(f"\nFirst frame shape: {first_frame.shape}")
    print("\nStarting segmentation workflow...")
    print("Follow the on-screen instructions:")
    print("  1. Place seeds (click to add, drag to move, click point to remove)")
    print("  2. Click 'Next Step' to run watershed and fit box")
    print("  3. Adjust box corners by dragging blue squares")
    print("  4. Click 'Next Step' to confirm box")
    print("  5. Click on left or right edge of Feature 1 to track")
    print("  6. Click 'Next Step' to fit curve model")
    
    # Run interactive workflow
    curve_model = run_segmentation_workflow(first_frame, 
                                           n_curve_points=args.n_points,
                                           box_fit_method=args.box_method)
    
    if curve_model is not None:
        print("\n✓ Curve model fitted successfully!")
        print(f"  Number of points: {curve_model.n_points}")
        print(f"  Spline type: {curve_model.spline_type}")
        print(f"  Curve length: {curve_model.get_total_length():.1f} pixels")
        print(f"  Backbone length: {curve_model.get_backbone_length():.1f} pixels")
        
        # Save curve model
        output_path = Path(args.video).stem + "_initial_curve.json"
        import json
        Path(output_path).write_text(json.dumps(curve_model.to_dict(), indent=2))
        print(f"\n✓ Initial curve saved to: {output_path}")
        print("\nNext steps:")
        print(f"  python main.py track {args.video} {output_path} --pz {args.pz}")
    else:
        print("\n✗ Segmentation cancelled or failed")


def cmd_track(args):
    """Run tracking workflow through video frames."""
    print("Tracking workflow not yet implemented.")
    print(f"Would track video: {args.video}")
    print(f"Using initial annotation: {args.annotation}")
    print(f"With pZ: {args.pz}")


def cmd_test_ui(args):
    """Test the UI with synthetic data."""
    import numpy as np
    import matplotlib.pyplot as plt
    from segmentvideo.models.curve import CurveModel
    
    print("Creating synthetic test frame...")
    
    # Create a simple test image
    test_image = np.random.rand(480, 640, 3) * 0.3 + 0.5
    
    # Add a curved feature
    y_coords = np.linspace(100, 380, 100)
    x_coords = 320 + 80 * np.sin(y_coords / 50)
    
    for x, y in zip(x_coords.astype(int), y_coords.astype(int)):
        test_image[y-5:y+5, x-5:x+5] = [0.2, 0.8, 0.2]
    
    # Create a test curve model
    curve_points = np.column_stack([x_coords[::10], y_coords[::10]])
    curve_model = CurveModel(frame_idx=0, points=curve_points)
    
    # Display
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(test_image)
    curve_model.render(ax, color='red', linewidth=3)
    ax.set_title("Test UI - Curve Model Rendering")
    plt.show()
    
    print("✓ Test UI displayed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="SegmentVideo - Assisted Manual Video Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segment command
    parser_segment = subparsers.add_parser('segment', 
        help='Run segmentation on first frame to create initial curve model')
    parser_segment.add_argument('video', help='Path to video file')
    parser_segment.add_argument('--pz', type=int, default=2,
        help='Temporal zoom level (default: 2)')
    parser_segment.add_argument('--n-points', type=int, default=10,
        help='Number of curve control points (default: 10)')
    parser_segment.add_argument('--box-method', type=str, default='ransac',
        choices=['ransac', 'pca', 'minarea'],
        help='Box fitting algorithm: ransac (robust to burrs/rounded corners), pca (clean rectangles), minarea (fastest) (default: ransac)')
    parser_segment.set_defaults(func=cmd_segment)
    
    # Track command
    parser_track = subparsers.add_parser('track',
        help='Track curve model through video frames')
    parser_track.add_argument('video', help='Path to video file')
    parser_track.add_argument('annotation', help='Path to initial annotation JSON')
    parser_track.add_argument('--pz', type=int, default=2,
        help='Temporal zoom level (default: 2)')
    parser_track.set_defaults(func=cmd_track)
    
    # Test UI command
    parser_test = subparsers.add_parser('test-ui',
        help='Test the UI with synthetic data')
    parser_test.set_defaults(func=cmd_test_ui)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
