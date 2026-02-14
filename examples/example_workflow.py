#!/usr/bin/env python3
"""
Example: Complete workflow from watershed to curve tracking.

This example demonstrates:
1. Loading a video
2. Running watershed segmentation on first frame
3. Fitting a curve model to a feature edge
4. Tracking through frames (placeholder)
"""

import numpy as np
from pathlib import Path

from segmentvideo.io import load_video_to_tensor, get_video_info
from segmentvideo.workflows import run_segmentation_workflow
from segmentvideo.annotation import AnnotationState


def example_workflow(video_path: str, pZ: int = 2, n_curve_points: int = 10):
    """
    Run complete workflow.
    
    Args:
        video_path: Path to video file
        pZ: Temporal zoom level
        n_curve_points: Number of curve control points
    """
    print("=" * 60)
    print("SegmentVideo - Complete Workflow Example")
    print("=" * 60)
    
    # Step 1: Load video
    print("\n[Step 1] Loading video...")
    print(f"  Video: {video_path}")
    print(f"  Temporal zoom (pZ): {pZ}")
    
    tensor, frame_indices = load_video_to_tensor(video_path, pZ=pZ)
    print(f"  Loaded tensor shape: {tensor.shape}")
    print(f"  Keyframes: {len(frame_indices)}")
    
    # Extract first frame
    first_frame_tensor = tensor[:, :, :, 0]
    first_frame = first_frame_tensor.transpose(2, 1, 0)  # (3, W, H) -> (H, W, 3)
    
    # Step 2: Run watershed segmentation
    print("\n[Step 2] Running interactive watershed segmentation...")
    print("  Instructions:")
    print("    1. Place seeds (click=add, drag=move, click point=remove)")
    print("    2. Click 'Update Mask' to run watershed")
    print("    3. Click on Feature 1 edge that you want to track")
    print("    4. Click 'Fit Curve' to extract curve model")
    
    curve_model = run_segmentation_workflow(first_frame, n_curve_points=n_curve_points)
    
    if curve_model is None:
        print("\n✗ Workflow cancelled")
        return
    
    print(f"\n✓ Curve model created!")
    print(f"  Points: {curve_model.n_points}")
    print(f"  Total length: {curve_model.get_total_length():.1f} px")
    print(f"  Backbone length: {curve_model.get_backbone_length():.1f} px")
    
    # Step 3: Initialize annotation state
    print("\n[Step 3] Initializing annotation state...")
    
    total_frames = int(get_video_info(video_path)['frames'])
    state = AnnotationState(total_frames=total_frames, frame_indices=frame_indices)
    
    # Set initial prediction for frame 0
    state.set_prediction(frame_indices[0], curve_model)
    state.verify_annotation(frame_indices[0])  # Mark as verified
    
    print(f"  Total frames: {total_frames}")
    print(f"  Keyframes: {len(frame_indices)}")
    
    state.print_progress()
    
    # Step 4: Save state
    output_path = Path(video_path).stem + "_annotations.json"
    state.save_to_file(output_path)
    
    print(f"\n✓ Annotations saved to: {output_path}")
    
    # Step 5: Tracking (placeholder)
    print("\n[Step 4] Tracking through frames...")
    print("  ⚠ Frame-by-frame tracking not yet implemented")
    print("  Next steps:")
    print(f"    - Load state from {output_path}")
    print("    - Iterate through keyframes")
    print("    - For each frame:")
    print("        * Use last verified model as starting point")
    print("        * Run model prediction")
    print("        * Show interactive UI for verification")
    print("        * Save corrected model")
    print("    - Interpolate between verified frames")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_workflow.py <video_file> [pZ] [n_points]")
        print()
        print("Arguments:")
        print("  video_file: Path to video file")
        print("  pZ: Temporal zoom (default: 2)")
        print("  n_points: Number of curve points (default: 10)")
        print()
        print("Example:")
        print("  python example_workflow.py ../videos/experiment.wmv 2 10")
        sys.exit(1)
    
    video_file = sys.argv[1]
    pZ = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_points = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    example_workflow(video_file, pZ=pZ, n_curve_points=n_points)
