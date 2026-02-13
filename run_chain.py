#!/usr/bin/env python3
"""
Run script for chain/curve tracking with N evenly-spaced segments.
Specialized for tracking linear features, filaments, or chains in videos.
"""

import numpy as np
from pathlib import Path
from lib.video import load_video_to_tensor, get_video_info
from lib.annotation import (
    AnnotationState,
    ModelParameters,
    AnnotationStatus,
    create_chain_params,
    verify_with_ui
)


def simple_chain_detector(frame: np.ndarray, n_segments: int, 
                          prev_params: ModelParameters = None) -> ModelParameters:
    """
    Simple chain detection using edge detection or other heuristics.
    
    This is a placeholder - replace with your actual detection logic.
    
    Args:
        frame: (3, W, H) tensor for a single frame
        n_segments: Number of segments in the chain
        prev_params: Parameters from previous frame (for temporal consistency)
        
    Returns:
        Predicted ModelParameters with chain points
    """
    W, H = frame.shape[1], frame.shape[2]
    
    if prev_params is not None and 'points' in prev_params.parameters:
        # Add small random motion from previous frame
        points = prev_params.parameters['points'].copy()
        points[:, :2] += np.random.randn(n_segments + 1, 2) * 2
        
        # Keep points in bounds
        points[:, 0] = np.clip(points[:, 0], 0, W - 1)
        points[:, 1] = np.clip(points[:, 1], 0, H - 1)
        
        return ModelParameters(
            frame_idx=0,
            model_type="curve",
            parameters={'points': points},
            status=AnnotationStatus.MODEL_PREDICTED
        )
    else:
        # Initialize with a diagonal line across the middle third of frame
        start_x, start_y = W // 3, H // 3
        end_x, end_y = 2 * W // 3, 2 * H // 3
        
        return create_chain_params(
            frame_idx=0,
            start_point=[start_x, start_y, 0],
            end_point=[end_x, end_y, 0],
            n_segments=n_segments,
            status=AnnotationStatus.MODEL_PREDICTED
        )


def chain_tracking_workflow(video_path: str, n_segments: int = 5, pZ: int = 2, 
                           use_interactive_ui: bool = True):
    """
    Complete chain tracking workflow with temporal consistency.
    
    Args:
        video_path: Path to video file
        n_segments: Number of segments in the chain
        pZ: Temporal zoom level (correlation time scale)
        use_interactive_ui: If True, use matplotlib UI for verification
    """
    print("="*80)
    print("CHAIN/CURVE TRACKING WORKFLOW")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of segments: {n_segments}")
    print(f"  Points per chain: {n_segments + 1}")
    print(f"  Temporal zoom (pZ): {pZ}")
    print(f"  Temporal consistency: ENABLED (uses last accepted state)")
    
    # Step 1: Load video at keyframe resolution
    print(f"\nStep 1: Loading video with pZ={pZ} (every {2**pZ} frames)")
    print("-"*80)
    tensor, frame_indices = load_video_to_tensor(video_path, pZ=pZ)
    
    # Initialize annotation state
    video_info = get_video_info(video_path)
    state = AnnotationState(video_info['frames'], frame_indices)
    
    print(f"\nLoaded {len(frame_indices)} keyframes from {video_info['frames']} total frames")
    
    # Step 2: Run model predictions on keyframes with temporal consistency
    print(f"\nStep 2: Running chain detection on keyframes (with temporal consistency)")
    print("-"*80)
    print("Note: Each prediction uses the last ACCEPTED state as initial estimate")
    
    last_accepted_params = None  # Will hold the last user-verified parameters
    
    for i, frame_idx in enumerate(frame_indices):
        frame = tensor[:, :, :, i]
        
        # Run chain detection using last accepted state
        predicted_params = simple_chain_detector(frame, n_segments, last_accepted_params)
        state.set_prediction(frame_idx, predicted_params)
        
        if (i + 1) % 50 == 0:
            print(f"  Predicted {i+1}/{len(frame_indices)} keyframes")
    
    print(f"Completed predictions for all {len(frame_indices)} keyframes")
    
    # Step 3: User verification with temporal consistency
    print(f"\nStep 3: User verification of keyframes")
    print("-"*80)
    if use_interactive_ui:
        print("Interactive UI enabled - drag control points to adjust chain shape")
        print("Each red dot is draggable. Press 'A' to accept, 'S' to skip")
        print("Temporal consistency: Each frame starts from your last accepted position")
    else:
        print("Auto-accept mode (no UI)")
    
    n_accepted = 0
    n_corrected = 0
    n_skipped = 0
    
    for i, frame_idx in enumerate(frame_indices):
        frame = tensor[:, :, :, i]
        predicted_params = state.get_annotation(frame_idx)
        
        # If we have a last accepted state, use it as the prediction base
        if last_accepted_params is not None:
            # Update prediction to start from last accepted position
            predicted_params.parameters['points'] = last_accepted_params.parameters['points'].copy()
            state.set_prediction(frame_idx, predicted_params)
        
        if use_interactive_ui:
            # Use interactive verification UI with drag support
            accepted, corrected_params = verify_with_ui(
                frame,
                predicted_params,
                frame_idx,
                tensor_idx=i,
                total_frames=len(frame_indices)
            )
            
            if accepted:
                # Check if parameters were modified
                params_changed = not np.allclose(
                    predicted_params.parameters.get('points', np.zeros((2, 3))),
                    corrected_params.parameters.get('points', np.zeros((2, 3)))
                )
                
                if params_changed:
                    n_corrected += 1
                else:
                    n_accepted += 1
                
                # Mark as verified
                state.verify_annotation(frame_idx, corrected_params)
                
                # Update last accepted state for next frame
                last_accepted_params = corrected_params.copy()
            else:
                n_skipped += 1
                print(f"  Skipped frame {frame_idx} - will use last accepted state for next frame")
        else:
            # Auto-accept mode
            state.verify_annotation(frame_idx, predicted_params)
            last_accepted_params = predicted_params.copy()
            n_accepted += 1
        
        if (i + 1) % 10 == 0 or i == len(frame_indices) - 1:
            print(f"  Progress: {i+1}/{len(frame_indices)} keyframes "
                  f"({n_accepted} accepted, {n_corrected} corrected, {n_skipped} skipped)")
    
    print(f"\nVerification complete:")
    print(f"  Accepted: {n_accepted} ({100*n_accepted/len(frame_indices):.1f}%)")
    print(f"  Corrected: {n_corrected} ({100*n_corrected/len(frame_indices):.1f}%)")
    print(f"  Skipped: {n_skipped} ({100*n_skipped/len(frame_indices):.1f}%)")
    
    # Step 4: Interpolate between verified keyframes
    print(f"\nStep 4: Interpolating between verified keyframes")
    print("-"*80)
    n_interpolated = state.interpolate_all_verified()
    print(f"Interpolated {n_interpolated} frames between {len(frame_indices)} keyframes")
    
    # Step 5: Show final progress
    print(f"\nStep 5: Final annotation statistics")
    print("-"*80)
    progress = state.get_progress()
    print(f"Total video frames: {progress['total_frames']}")
    print(f"Keyframes (pZ={pZ}): {progress['n_keyframes']}")
    print(f"User-verified keyframes: {progress['n_verified']}")
    print(f"Interpolated frames: {progress['n_interpolated']}")
    print(f"Total annotated frames: {progress['n_annotated']}")
    print(f"Coverage: {100*progress['total_progress']:.1f}%")
    
    # Calculate time savings
    manual_time_per_frame = 45  # seconds (chains are harder than ellipsoids)
    assisted_time_per_frame = 10  # seconds (verification with drag)
    assisted_with_temporal = 5   # seconds (with temporal consistency, less adjustment needed)
    
    manual_total_time = progress['total_frames'] * manual_time_per_frame
    assisted_total_time = progress['n_keyframes'] * assisted_with_temporal
    time_saved = manual_total_time - assisted_total_time
    
    print(f"\nEstimated time savings (with temporal consistency):")
    print(f"  Pure manual: {manual_total_time/3600:.1f} hours")
    print(f"  Assisted (verify keyframes): {assisted_total_time/3600:.1f} hours")
    print(f"  Time saved: {time_saved/3600:.1f} hours ({100*time_saved/manual_total_time:.1f}%)")
    print(f"  Temporal consistency reduces adjustment time by ~50%!")
    
    return state, tensor, frame_indices


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_chain.py <video_file> [n_segments] [pZ] [--no-ui]")
        print("\nTrack chains/curves with N evenly-spaced segments.")
        print("\nArguments:")
        print("  video_file: Path to video file")
        print("  n_segments: Number of segments (default=5, creates 6 points)")
        print("  pZ: Temporal zoom (default=2)")
        print("  --no-ui: Disable interactive UI (auto-accept all predictions)")
        print("\nExamples:")
        print("  python run_chain.py video.wmv 10 2       # Track 10-segment chain")
        print("  python run_chain.py video.wmv 5 3        # 5 segments, higher pZ")
        print("  python run_chain.py video.wmv 8 2 --no-ui  # Auto-accept mode")
        sys.exit(1)
    
    video_file = sys.argv[1]
    n_segments = 5
    pZ = 2
    use_ui = True
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--no-ui':
            use_ui = False
        else:
            try:
                val = int(arg)
                if i == 2:
                    n_segments = val
                elif i == 3:
                    pZ = val
            except ValueError:
                print(f"Warning: ignoring unrecognized argument: {arg}")
    
    print(f"Configuration: {n_segments} segments, pZ={pZ}, UI={'enabled' if use_ui else 'disabled'}")
    
    # Run the workflow
    state, tensor, frame_indices = chain_tracking_workflow(
        video_file, 
        n_segments=n_segments,
        pZ=pZ, 
        use_interactive_ui=use_ui
    )
    
    print("\n" + "="*80)
    print("CHAIN TRACKING COMPLETE!")
    print("="*80)
    print(f"Tracked {n_segments+1} points across {len(frame_indices)} keyframes")
