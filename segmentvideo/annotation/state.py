"""
Annotation state management for tracking models through video frames.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
from enum import Enum

from segmentvideo.models.base import BaseModel


class AnnotationStatus(Enum):
    """Status of an annotation for a frame."""
    EMPTY = "empty"  # No annotation
    PREDICTED = "predicted"  # Model prediction, not verified
    VERIFIED = "verified"  # User verified (with or without corrections)
    INTERPOLATED = "interpolated"  # Interpolated from verified keyframes


class FrameAnnotation:
    """Represents the annotation for a single frame."""
    
    def __init__(self, frame_idx: int):
        self.frame_idx = frame_idx
        self.status = AnnotationStatus.EMPTY
        self.model: Optional[BaseModel] = None
        self.previous_model: Optional[BaseModel] = None  # For undo functionality
    
    def set_prediction(self, model: BaseModel):
        """Set model prediction."""
        self.model = model
        self.status = AnnotationStatus.PREDICTED
    
    def verify(self, corrected_model: Optional[BaseModel] = None):
        """
        Mark annotation as verified.
        
        Args:
            corrected_model: If provided, use this instead of current model
        """
        if corrected_model is not None:
            self.previous_model = self.model
            self.model = corrected_model
        
        self.status = AnnotationStatus.VERIFIED
    
    def set_interpolated(self, model: BaseModel):
        """Set interpolated model."""
        self.model = model
        self.status = AnnotationStatus.INTERPOLATED
    
    def is_verified(self) -> bool:
        """Check if this annotation is verified."""
        return self.status == AnnotationStatus.VERIFIED
    
    def is_empty(self) -> bool:
        """Check if this annotation is empty."""
        return self.status == AnnotationStatus.EMPTY
    
    def undo(self):
        """Undo last verification (restore previous model)."""
        if self.previous_model is not None:
            self.model = self.previous_model
            self.previous_model = None
            self.status = AnnotationStatus.PREDICTED


class AnnotationState:
    """
    Manages the annotation state for all frames in a video.
    
    Handles:
    - Storing predictions and verifications
    - Temporal consistency (using last verified frame as starting point)
    - Interpolation between keyframes
    - Progress tracking
    """
    
    def __init__(self, total_frames: int, frame_indices: Optional[np.ndarray] = None):
        """
        Initialize annotation state.
        
        Args:
            total_frames: Total number of frames in the original video
            frame_indices: Optional array mapping downsampled indices to original frame numbers
        """
        self.total_frames = total_frames
        self.frame_indices = frame_indices if frame_indices is not None else np.arange(total_frames)
        self.n_keyframes = len(self.frame_indices)
        
        # Store annotations for each keyframe
        self.annotations: Dict[int, FrameAnnotation] = {}
        
        # Initialize all frames as empty
        for frame_idx in self.frame_indices:
            self.annotations[frame_idx] = FrameAnnotation(frame_idx)
    
    def set_prediction(self, frame_idx: int, model: BaseModel):
        """
        Store a model prediction for a frame.
        
        Args:
            frame_idx: Original frame index
            model: Model prediction
        """
        if frame_idx not in self.annotations:
            raise ValueError(f"Frame index {frame_idx} not in annotations")
        
        self.annotations[frame_idx].set_prediction(model)
    
    def verify_annotation(self, frame_idx: int, corrected_model: Optional[BaseModel] = None):
        """
        Verify an annotation (mark as accepted by user).
        
        Args:
            frame_idx: Original frame index
            corrected_model: If provided, use this corrected model instead of prediction
        """
        if frame_idx not in self.annotations:
            raise ValueError(f"Frame index {frame_idx} not in annotations")
        
        self.annotations[frame_idx].verify(corrected_model)
    
    def get_annotation(self, frame_idx: int) -> Optional[FrameAnnotation]:
        """Get annotation for a specific frame."""
        return self.annotations.get(frame_idx)
    
    def get_model(self, frame_idx: int) -> Optional[BaseModel]:
        """Get the model for a specific frame."""
        annotation = self.get_annotation(frame_idx)
        return annotation.model if annotation else None
    
    def get_last_verified_frame(self, before_frame: int) -> Optional[int]:
        """
        Get the most recent verified frame before the given frame.
        
        This is used for temporal consistency: the next prediction starts
        from the last verified position.
        
        Args:
            before_frame: Frame index
            
        Returns:
            Frame index of last verified frame, or None if no verified frames exist
        """
        verified_frames = [
            idx for idx in self.frame_indices 
            if idx < before_frame and self.annotations[idx].is_verified()
        ]
        
        return max(verified_frames) if verified_frames else None
    
    def get_last_verified_model(self, before_frame: int) -> Optional[BaseModel]:
        """
        Get the model from the last verified frame before the given frame.
        
        Returns:
            BaseModel instance or None
        """
        last_frame = self.get_last_verified_frame(before_frame)
        return self.get_model(last_frame) if last_frame is not None else None
    
    def interpolate_between_keyframes(self, frame1: int, frame2: int):
        """
        Interpolate annotations between two verified keyframes.
        
        Both keyframes must be verified.
        
        Args:
            frame1: Earlier keyframe index
            frame2: Later keyframe index
        """
        if frame1 not in self.annotations or frame2 not in self.annotations:
            raise ValueError(f"Invalid keyframe indices: {frame1}, {frame2}")
        
        annot1 = self.annotations[frame1]
        annot2 = self.annotations[frame2]
        
        if not annot1.is_verified() or not annot2.is_verified():
            raise ValueError(f"Both keyframes must be verified for interpolation")
        
        model1 = annot1.model
        model2 = annot2.model
        
        # Find all frames between frame1 and frame2
        frames_between = [
            idx for idx in self.frame_indices
            if frame1 < idx < frame2
        ]
        
        # Interpolate each frame
        for frame_idx in frames_between:
            # Calculate interpolation parameter
            alpha = (frame_idx - frame1) / (frame2 - frame1)
            
            # Interpolate model
            interpolated_model = model1.interpolate(model2, alpha)
            interpolated_model.frame_idx = frame_idx
            
            # Store as interpolated
            self.annotations[frame_idx].set_interpolated(interpolated_model)
    
    def interpolate_all_verified(self):
        """
        Interpolate between all pairs of consecutive verified keyframes.
        """
        # Get all verified frame indices in order
        verified_frames = sorted([
            idx for idx in self.frame_indices
            if self.annotations[idx].is_verified()
        ])
        
        if len(verified_frames) < 2:
            print("Need at least 2 verified frames for interpolation")
            return
        
        # Interpolate between each consecutive pair
        interpolated_count = 0
        for i in range(len(verified_frames) - 1):
            frame1 = verified_frames[i]
            frame2 = verified_frames[i + 1]
            
            # Count frames that will be interpolated
            frames_between = [
                idx for idx in self.frame_indices
                if frame1 < idx < frame2
            ]
            
            if frames_between:
                self.interpolate_between_keyframes(frame1, frame2)
                interpolated_count += len(frames_between)
        
        print(f"Interpolated {interpolated_count} frames between {len(verified_frames)} verified keyframes")
    
    def get_progress(self) -> Dict[str, int]:
        """
        Get annotation progress statistics.
        
        Returns:
            Dictionary with progress information
        """
        stats = {
            'total_frames': self.total_frames,
            'n_keyframes': self.n_keyframes,
            'n_empty': 0,
            'n_predicted': 0,
            'n_verified': 0,
            'n_interpolated': 0
        }
        
        for annotation in self.annotations.values():
            if annotation.status == AnnotationStatus.EMPTY:
                stats['n_empty'] += 1
            elif annotation.status == AnnotationStatus.PREDICTED:
                stats['n_predicted'] += 1
            elif annotation.status == AnnotationStatus.VERIFIED:
                stats['n_verified'] += 1
            elif annotation.status == AnnotationStatus.INTERPOLATED:
                stats['n_interpolated'] += 1
        
        stats['n_annotated'] = stats['n_verified'] + stats['n_interpolated']
        stats['progress_pct'] = 100 * stats['n_annotated'] / self.n_keyframes
        
        return stats
    
    def print_progress(self):
        """Print progress statistics."""
        stats = self.get_progress()
        
        print("\n=== Annotation Progress ===")
        print(f"Keyframes: {stats['n_keyframes']}/{stats['total_frames']} (temporal downsampling)")
        print(f"Verified: {stats['n_verified']}")
        print(f"Interpolated: {stats['n_interpolated']}")
        print(f"Predicted (not verified): {stats['n_predicted']}")
        print(f"Empty: {stats['n_empty']}")
        print(f"Total annotated: {stats['n_annotated']}/{stats['n_keyframes']} ({stats['progress_pct']:.1f}%)")
    
    def export_annotations(self) -> List[Dict]:
        """
        Export all annotations as a list of dictionaries.
        
        Returns:
            List of annotation dictionaries
        """
        export_data = []
        
        for frame_idx in sorted(self.frame_indices):
            annotation = self.annotations[frame_idx]
            
            if annotation.model is not None:
                export_data.append({
                    'frame_idx': frame_idx,
                    'status': annotation.status.value,
                    'model': annotation.model.to_dict()
                })
        
        return export_data
    
    def save_to_file(self, filepath: str):
        """
        Save annotations to a JSON file.
        
        Args:
            filepath: Path to output file
        """
        import json
        from pathlib import Path
        
        data = {
            'total_frames': self.total_frames,
            'frame_indices': self.frame_indices.tolist(),
            'annotations': self.export_annotations(),
            'progress': self.get_progress()
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
        print(f"Annotations saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'AnnotationState':
        """
        Load annotations from a JSON file.
        
        Args:
            filepath: Path to input file
            
        Returns:
            AnnotationState instance
        """
        import json
        from pathlib import Path
        from segmentvideo.models.curve import CurveModel
        from segmentvideo.models.ellipsoid import EllipsoidModel
        
        data = json.loads(Path(filepath).read_text())
        
        # Create state
        state = cls(
            total_frames=data['total_frames'],
            frame_indices=np.array(data['frame_indices'])
        )
        
        # Load annotations
        for annot_data in data['annotations']:
            frame_idx = annot_data['frame_idx']
            status_str = annot_data['status']
            model_data = annot_data['model']
            
            # Reconstruct model based on type
            model_type = model_data['model_type']
            if model_type == 'CurveModel':
                model = CurveModel.from_dict(model_data)
            elif model_type == 'EllipsoidModel':
                model = EllipsoidModel.from_dict(model_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Set annotation
            annotation = state.annotations[frame_idx]
            annotation.model = model
            annotation.status = AnnotationStatus(status_str)
        
        print(f"Annotations loaded from {filepath}")
        state.print_progress()
        
        return state
