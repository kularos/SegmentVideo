#!/usr/bin/env python3
"""
Annotation state management for assisted manual segmentation.
Tracks user-verified model parameters and interpolates between keyframes.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AnnotationStatus(Enum):
    """Status of frame annotation."""
    UNPROCESSED = "unprocessed"  # Not yet shown to user
    MODEL_PREDICTED = "predicted"  # Model has made a prediction
    USER_VERIFIED = "verified"  # User has confirmed/corrected the annotation
    INTERPOLATED = "interpolated"  # Computed from verified keyframes


@dataclass
class ModelParameters:
    """
    Generic container for model parameters at a single frame.

    Can represent different primitive models:
    - Ellipsoid: center (x,y,z), semi-axes (a,b,c), rotation (roll, pitch, yaw)
    - Curve points: N points [(x1,y1,z1), (x2,y2,z2), ...]
    - Multi-object: list of sub-models
    """
    frame_idx: int  # Original video frame index
    model_type: str  # e.g., "ellipsoid", "curve", "multi_ellipsoid"
    parameters: Dict[str, np.ndarray]  # Flexible parameter storage
    confidence: float = 1.0  # Confidence score (for predictions)
    status: AnnotationStatus = AnnotationStatus.UNPROCESSED

    def copy(self) -> 'ModelParameters':
        """Create a deep copy of the model parameters."""
        return ModelParameters(
            frame_idx=self.frame_idx,
            model_type=self.model_type,
            parameters={k: v.copy() for k, v in self.parameters.items()},
            confidence=self.confidence,
            status=self.status
        )


class AnnotationState:
    """
    Manages the state of annotations across all frames.

    Workflow:
    1. Load keyframes at temporal downsampling level (pZ)
    2. Model predicts on keyframes
    3. User verifies/corrects keyframe annotations
    4. Interpolate between verified keyframes to fill dense frames
    """

    def __init__(self, total_frames: int, keyframe_indices: np.ndarray):
        """
        Initialize annotation state.

        Args:
            total_frames: Total number of frames in original video
            keyframe_indices: Original frame indices of loaded keyframes (from pZ downsampling)
        """
        self.total_frames = total_frames
        self.keyframe_indices = keyframe_indices
        self.n_keyframes = len(keyframe_indices)

        # Map from original frame index to ModelParameters
        self.annotations: Dict[int, ModelParameters] = {}

        # Track which keyframes have been verified
        self.verified_keyframes: List[int] = []

    def set_prediction(self, frame_idx: int, model_params: ModelParameters):
        """Store a model prediction for a keyframe."""
        model_params.frame_idx = frame_idx
        model_params.status = AnnotationStatus.MODEL_PREDICTED
        self.annotations[frame_idx] = model_params

    def verify_annotation(self, frame_idx: int, model_params: Optional[ModelParameters] = None):
        """
        Mark a keyframe as user-verified.

        Args:
            frame_idx: Original frame index
            model_params: Updated parameters if user made corrections, None to keep existing
        """
        if model_params is not None:
            model_params.frame_idx = frame_idx
            model_params.status = AnnotationStatus.USER_VERIFIED
            self.annotations[frame_idx] = model_params
        elif frame_idx in self.annotations:
            self.annotations[frame_idx].status = AnnotationStatus.USER_VERIFIED
        else:
            raise ValueError(f"Cannot verify frame {frame_idx}: no annotation exists")

        if frame_idx not in self.verified_keyframes:
            self.verified_keyframes.append(frame_idx)
            self.verified_keyframes.sort()

    def get_annotation(self, frame_idx: int) -> Optional[ModelParameters]:
        """Retrieve annotation for a specific frame."""
        return self.annotations.get(frame_idx, None)

    def get_verified_keyframes(self) -> List[int]:
        """Get list of frame indices that have been user-verified."""
        return self.verified_keyframes.copy()

    def interpolate_between_keyframes(self, start_idx: int, end_idx: int) -> Dict[int, ModelParameters]:
        """
        Interpolate model parameters between two verified keyframes.

        Args:
            start_idx: Frame index of first verified keyframe
            end_idx: Frame index of second verified keyframe

        Returns:
            Dictionary mapping frame indices to interpolated parameters
        """
        if start_idx not in self.annotations or end_idx not in self.annotations:
            raise ValueError(f"Cannot interpolate: missing annotations at {start_idx} or {end_idx}")

        start_params = self.annotations[start_idx]
        end_params = self.annotations[end_idx]

        if start_params.model_type != end_params.model_type:
            raise ValueError(
                f"Cannot interpolate between different model types: {start_params.model_type} vs {end_params.model_type}")

        # Linear interpolation between frames
        interpolated = {}
        for frame_idx in range(start_idx + 1, end_idx):
            # Interpolation weight: 0 at start, 1 at end
            alpha = (frame_idx - start_idx) / (end_idx - start_idx)

            # Interpolate each parameter
            interp_params = {}
            for key in start_params.parameters.keys():
                start_val = start_params.parameters[key]
                end_val = end_params.parameters[key]

                # Linear interpolation
                interp_val = (1 - alpha) * start_val + alpha * end_val
                interp_params[key] = interp_val

            model = ModelParameters(
                frame_idx=frame_idx,
                model_type=start_params.model_type,
                parameters=interp_params,
                confidence=min(start_params.confidence, end_params.confidence),
                status=AnnotationStatus.INTERPOLATED
            )
            interpolated[frame_idx] = model
            self.annotations[frame_idx] = model

        return interpolated

    def interpolate_all_verified(self) -> int:
        """
        Interpolate between all consecutive verified keyframes.

        Returns:
            Number of frames interpolated
        """
        if len(self.verified_keyframes) < 2:
            return 0

        total_interpolated = 0
        for i in range(len(self.verified_keyframes) - 1):
            start = self.verified_keyframes[i]
            end = self.verified_keyframes[i + 1]
            interpolated = self.interpolate_between_keyframes(start, end)
            total_interpolated += len(interpolated)

        return total_interpolated

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current annotation progress statistics.

        Returns:
            Dictionary with progress metrics
        """
        n_verified = len(self.verified_keyframes)
        n_predicted = sum(1 for a in self.annotations.values()
                          if a.status == AnnotationStatus.MODEL_PREDICTED)
        n_interpolated = sum(1 for a in self.annotations.values()
                             if a.status == AnnotationStatus.INTERPOLATED)

        return {
            'total_frames': self.total_frames,
            'n_keyframes': self.n_keyframes,
            'n_verified': n_verified,
            'n_predicted': n_predicted,
            'n_interpolated': n_interpolated,
            'n_annotated': len(self.annotations),
            'keyframe_progress': n_verified / self.n_keyframes if self.n_keyframes > 0 else 0,
            'total_progress': len(self.annotations) / self.total_frames if self.total_frames > 0 else 0
        }

    def export_annotations(self) -> Dict[str, Any]:
        """
        Export all annotations in a serializable format.

        Returns:
            Dictionary containing all annotation data
        """
        return {
            'total_frames': self.total_frames,
            'keyframe_indices': self.keyframe_indices.tolist(),
            'annotations': {
                frame_idx: {
                    'model_type': params.model_type,
                    'parameters': {k: v.tolist() for k, v in params.parameters.items()},
                    'confidence': params.confidence,
                    'status': params.status.value
                }
                for frame_idx, params in self.annotations.items()
            }
        }


# Example helper functions for common primitive models

def create_ellipsoid_params(frame_idx: int, center: np.ndarray, semi_axes: np.ndarray,
                            rotation: np.ndarray,
                            status: AnnotationStatus = AnnotationStatus.MODEL_PREDICTED) -> ModelParameters:
    """
    Create ModelParameters for a 3D ellipsoid.

    Args:
        frame_idx: Frame index
        center: [x, y, z] center position
        semi_axes: [a, b, c] semi-axis lengths
        rotation: [roll, pitch, yaw] rotation angles in radians
        status: Annotation status
    """
    return ModelParameters(
        frame_idx=frame_idx,
        model_type="ellipsoid",
        parameters={
            'center': np.array(center, dtype=np.float32),
            'semi_axes': np.array(semi_axes, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32)
        },
        status=status
    )


def create_curve_params(frame_idx: int, points: np.ndarray,
                        status: AnnotationStatus = AnnotationStatus.MODEL_PREDICTED) -> ModelParameters:
    """
    Create ModelParameters for a curve defined by points.

    Args:
        frame_idx: Frame index
        points: (N, 3) array of [x, y, z] points
        status: Annotation status
    """
    return ModelParameters(
        frame_idx=frame_idx,
        model_type="curve",
        parameters={
            'points': np.array(points, dtype=np.float32)
        },
        status=status
    )


if __name__ == "__main__":
    # Example usage
    print("Example: Tracking an ellipsoid through 1000 frames with pZ=2")
    print("-" * 60)

    # Simulate pZ=2: every 4th frame is a keyframe
    total_frames = 1000
    keyframe_indices = np.arange(0, total_frames, 2)

    state = AnnotationState(total_frames, keyframe_indices)

    # Simulate model predictions on first few keyframes
    for i, frame_idx in enumerate(keyframe_indices[:5]):
        center = np.array([100 + i * 10, 200 + i * 5, 50])
        semi_axes = np.array([20, 15, 10])
        rotation = np.array([0, 0, i * 0.1])

        params = create_ellipsoid_params(frame_idx, center, semi_axes, rotation)
        state.set_prediction(frame_idx, params)

    print(f"Made predictions for {len(keyframe_indices[:5])} keyframes")

    # User verifies first and last
    state.verify_annotation(keyframe_indices[0])
    state.verify_annotation(keyframe_indices[4])

    print(f"User verified 2 keyframes")

    # Interpolate between verified keyframes
    n_interpolated = state.interpolate_all_verified()
    print(f"Interpolated {n_interpolated} intermediate frames")

    # Check progress
    progress = state.get_progress()
    print(f"\nProgress:")
    print(
        f"  Keyframes verified: {progress['n_verified']}/{progress['n_keyframes']} ({100 * progress['keyframe_progress']:.1f}%)")
    print(
        f"  Total frames annotated: {progress['n_annotated']}/{progress['total_frames']} ({100 * progress['total_progress']:.1f}%)")