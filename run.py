#!/usr/bin/env python3
"""
Example workflow for assisted manual segmentation.

This demonstrates the complete pipeline:
1. Load video at correlation time scale (pZ)
2. Run model predictions on keyframes
3. User verifies/corrects predictions
4. Interpolate to fill all frames
"""

import numpy as np
from lib.video import load_video_to_tensor, get_video_info
from lib.annotation import (
    AnnotationState,
    ModelParameters,
    AnnotationStatus,
    create_ellipsoid_params
)


def dummy_segmentation_model(frame: np.ndarray, prev_params: ModelParameters = None) -> ModelParameters:
    """
    Placeholder for your actual segmentation model.

    This would be replaced with your primitive model fitting logic
    (e.g., ellipsoid fitting, curve extraction, etc.)

    Args:
        frame: (3, W, H) tensor for a single frame
        prev_params: Parameters from previous frame (for temporal consistency)

    Returns:
        Predicted ModelParameters
    """
    # Dummy prediction: just return a fixed ellipsoid with some noise
    center = np.array([frame.shape[1] // 2, frame.shape[2] // 2, 100], dtype=np.float32)
    if prev_params is not None and 'center' in prev_params.parameters:
        # Add small random motion from previous frame
        center = prev_params.parameters['center'] + np.random.randn(3) * 2

    semi_axes = np.array([30, 20, 15], dtype=np.float32)
    rotation = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    return create_ellipsoid_params(
        frame_idx=0,  # Will be set by AnnotationState
        center=center,
        semi_axes=semi_axes,
        rotation=rotation,
        status=AnnotationStatus.MODEL_PREDICTED
    )


def user_verification_interface(frame: np.ndarray, predicted_params: ModelParameters) -> tuple[bool, ModelParameters]:
    """
    Placeholder for user verification interface.

    In a real application, this would:
    - Display the frame with the predicted model overlay
    - Allow user to adjust model parameters interactively
    - Return whether user accepted or modified the prediction

    Args:
        frame: (3, W, H) tensor for the frame
        predicted_params: Model's prediction

    Returns:
        (accepted, corrected_params): Whether user accepted, and final parameters
    """
    # Dummy: randomly "accept" 80% of predictions, "correct" 20%
    accepted = np.random.rand() > 0.2

    if accepted:
        return True, predicted_params
    else:
        # Simulate user making small corrections
        corrected = predicted_params.copy()
        if 'center' in corrected.parameters:
            corrected.parameters['center'] += np.random.randn(3) * 5
        return False, corrected


def assisted_segmentation_workflow(video_path: str, pZ: int = 2):
    """
    Complete assisted segmentation workflow.

    Args:
        video_path: Path to video file
        pZ: Temporal zoom level (correlation time scale)
    """
    print("=" * 80)
    print("ASSISTED MANUAL SEGMENTATION WORKFLOW")
    print("=" * 80)

    # Step 1: Load video at keyframe resolution
    print(f"\nStep 1: Loading video with pZ={pZ} (every {2 ** pZ} frames)")
    print("-" * 80)
    tensor, frame_indices = load_video_to_tensor(video_path, pZ=pZ)

    # Initialize annotation state
    video_info = get_video_info(video_path)
    state = AnnotationState(video_info['frames'], frame_indices)

    print(f"\nLoaded {len(frame_indices)} keyframes from {video_info['frames']} total frames")

    # Step 2: Run model predictions on keyframes
    print(f"\nStep 2: Running model predictions on keyframes")
    print("-" * 80)
    prev_params = None
    for i, frame_idx in enumerate(frame_indices):
        frame = tensor[:, :, :, i]

        # Run model prediction
        predicted_params = dummy_segmentation_model(frame, prev_params)
        state.set_prediction(frame_idx, predicted_params)

        prev_params = predicted_params

        if (i + 1) % 50 == 0:
            print(f"  Predicted {i + 1}/{len(frame_indices)} keyframes")

    print(f"Completed predictions for all {len(frame_indices)} keyframes")

    # Step 3: User verification (simulated)
    print(f"\nStep 3: User verification of keyframes")
    print("-" * 80)
    n_accepted = 0
    n_corrected = 0

    for i, frame_idx in enumerate(frame_indices):
        frame = tensor[:, :, :, i]
        predicted_params = state.get_annotation(frame_idx)

        # Simulate user verification
        accepted, corrected_params = user_verification_interface(frame, predicted_params)

        # Mark as verified
        state.verify_annotation(frame_idx, corrected_params)

        if accepted:
            n_accepted += 1
        else:
            n_corrected += 1

        if (i + 1) % 50 == 0:
            print(f"  Verified {i + 1}/{len(frame_indices)} keyframes ({n_accepted} accepted, {n_corrected} corrected)")

    print(f"\nVerification complete:")
    print(f"  Accepted: {n_accepted} ({100 * n_accepted / len(frame_indices):.1f}%)")
    print(f"  Corrected: {n_corrected} ({100 * n_corrected / len(frame_indices):.1f}%)")

    # Step 4: Interpolate between verified keyframes
    print(f"\nStep 4: Interpolating between verified keyframes")
    print("-" * 80)
    n_interpolated = state.interpolate_all_verified()
    print(f"Interpolated {n_interpolated} frames between {len(frame_indices)} keyframes")

    # Step 5: Show final progress
    print(f"\nStep 5: Final annotation statistics")
    print("-" * 80)
    progress = state.get_progress()
    print(f"Total video frames: {progress['total_frames']}")
    print(f"Keyframes (pZ={pZ}): {progress['n_keyframes']}")
    print(f"User-verified keyframes: {progress['n_verified']}")
    print(f"Interpolated frames: {progress['n_interpolated']}")
    print(f"Total annotated frames: {progress['n_annotated']}")
    print(f"Coverage: {100 * progress['total_progress']:.1f}%")

    # Calculate time savings
    manual_time_per_frame = 30  # seconds (example)
    assisted_time_per_frame = 5  # seconds (just verification)

    manual_total_time = progress['total_frames'] * manual_time_per_frame
    assisted_total_time = progress['n_keyframes'] * assisted_time_per_frame
    time_saved = manual_total_time - assisted_total_time

    print(f"\nEstimated time savings:")
    print(f"  Pure manual: {manual_total_time / 3600:.1f} hours")
    print(f"  Assisted (verify keyframes): {assisted_total_time / 3600:.1f} hours")
    print(f"  Time saved: {time_saved / 3600:.1f} hours ({100 * time_saved / manual_total_time:.1f}%)")

    # Export annotations
    print(f"\nStep 6: Exporting annotations")
    print("-" * 80)
    annotations = state.export_annotations()
    print(f"Exported {len(annotations['annotations'])} frame annotations")

    return state, tensor, frame_indices


if __name__ == "__main__":



    video_file = f"videos/200Volts-5cycles.wmv"
    pZ = 4
    state, tensor, frame_indices = assisted_segmentation_workflow(video_file, pZ)

    print("\n" + "=" * 80)
    print("Workflow complete! Annotations are ready for export.")
    print("=" * 80)