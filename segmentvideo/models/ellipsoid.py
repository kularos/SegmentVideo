"""
Ellipsoid model for tracking 3D ellipsoidal features.
"""

from typing import List, Dict, Any
import numpy as np
from .base import BaseModel, ControlPoint


class EllipsoidModel(BaseModel):
    """
    3D Ellipsoid model with center, semi-axes, and rotation.
    
    Displayed as a 2D ellipse projection in the UI.
    
    Parameters:
        - center: [x, y, z] position
        - semi_axes: [a, b, c] semi-axis lengths
        - rotation: [roll, pitch, yaw] rotation angles in radians
    """
    
    def __init__(self, frame_idx: int, center: np.ndarray, semi_axes: np.ndarray, 
                 rotation: np.ndarray):
        """
        Initialize ellipsoid model.
        
        Args:
            frame_idx: Frame index
            center: Center position [x, y, z]
            semi_axes: Semi-axis lengths [a, b, c]
            rotation: Rotation angles [roll, pitch, yaw] in radians
        """
        super().__init__(frame_idx)
        self.center = np.array(center, dtype=np.float32)
        self.semi_axes = np.array(semi_axes, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        
        if self.center.shape != (3,):
            raise ValueError(f"Center must be (3,), got {self.center.shape}")
        if self.semi_axes.shape != (3,):
            raise ValueError(f"Semi-axes must be (3,), got {self.semi_axes.shape}")
        if self.rotation.shape != (3,):
            raise ValueError(f"Rotation must be (3,), got {self.rotation.shape}")
    
    @classmethod
    def from_2d(cls, frame_idx: int, center_x: float, center_y: float, 
                width: float, height: float, angle: float = 0.0) -> 'EllipsoidModel':
        """
        Create ellipsoid from 2D ellipse parameters (z=0, no 3D rotation).
        
        Args:
            frame_idx: Frame index
            center_x, center_y: Center position
            width, height: Semi-axis lengths (a, b)
            angle: Rotation angle in radians (rotation around z-axis)
            
        Returns:
            EllipsoidModel instance
        """
        return cls(
            frame_idx=frame_idx,
            center=np.array([center_x, center_y, 0.0]),
            semi_axes=np.array([width, height, min(width, height)]),
            rotation=np.array([0.0, 0.0, angle])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'model_type': 'EllipsoidModel',
            'frame_idx': self.frame_idx,
            'center': self.center.tolist(),
            'semi_axes': self.semi_axes.tolist(),
            'rotation': self.rotation.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EllipsoidModel':
        """Deserialize from dictionary."""
        return cls(
            frame_idx=data['frame_idx'],
            center=np.array(data['center']),
            semi_axes=np.array(data['semi_axes']),
            rotation=np.array(data['rotation'])
        )
    
    def interpolate(self, other: 'EllipsoidModel', alpha: float) -> 'EllipsoidModel':
        """
        Interpolate between this ellipsoid and another.
        
        Uses linear interpolation for all parameters.
        For rotation, uses linear interpolation of angles (simple but works for small angles).
        
        Args:
            other: Another EllipsoidModel
            alpha: Interpolation factor (0.0 = this, 1.0 = other)
            
        Returns:
            New EllipsoidModel with interpolated parameters
        """
        if not isinstance(other, EllipsoidModel):
            raise TypeError(f"Cannot interpolate EllipsoidModel with {type(other)}")
        
        # Linear interpolation
        interpolated_center = (1 - alpha) * self.center + alpha * other.center
        interpolated_semi_axes = (1 - alpha) * self.semi_axes + alpha * other.semi_axes
        
        # For rotation, we need to handle angle wrapping
        # Simple linear interpolation (works for small angle changes)
        interpolated_rotation = self._interpolate_angles(self.rotation, other.rotation, alpha)
        
        # Calculate interpolated frame index
        interpolated_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)
        
        return EllipsoidModel(
            frame_idx=interpolated_frame_idx,
            center=interpolated_center,
            semi_axes=interpolated_semi_axes,
            rotation=interpolated_rotation
        )
    
    def _interpolate_angles(self, angle1: np.ndarray, angle2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Interpolate angles with proper wrapping.
        
        Handles the discontinuity at ±π.
        """
        # Normalize angles to [-π, π]
        angle1_norm = np.arctan2(np.sin(angle1), np.cos(angle1))
        angle2_norm = np.arctan2(np.sin(angle2), np.cos(angle2))
        
        # Calculate shortest angular distance
        diff = angle2_norm - angle1_norm
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        
        # Interpolate
        interpolated = angle1_norm + alpha * diff
        
        return interpolated
    
    def render(self, ax, color='red', linewidth=2, show_center=True, 
               show_axes=True, alpha=1.0, **kwargs):
        """
        Render the ellipsoid as a 2D ellipse projection.
        
        Args:
            ax: Matplotlib axes
            color: Color for the ellipse
            linewidth: Line width
            show_center: Whether to show center point
            show_axes: Whether to show major/minor axes
            alpha: Transparency
        """
        from matplotlib.patches import Ellipse
        
        # For 2D display, we project the 3D ellipsoid to 2D
        # Use XY plane projection (ignore Z for now)
        center_2d = self.center[:2]  # [x, y]
        
        # For 2D ellipse, use first two semi-axes
        width = 2 * self.semi_axes[0]  # Full width (2*a)
        height = 2 * self.semi_axes[1]  # Full height (2*b)
        
        # Rotation angle (yaw, rotation around Z-axis)
        angle_deg = np.degrees(self.rotation[2])
        
        # Draw ellipse
        ellipse = Ellipse(
            xy=center_2d,
            width=width,
            height=height,
            angle=angle_deg,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            **kwargs
        )
        ax.add_patch(ellipse)
        
        if show_center:
            # Draw center point
            ax.plot(center_2d[0], center_2d[1], 'o', 
                   color=color, markersize=8, markerfacecolor=color)
        
        if show_axes:
            # Draw major axis
            angle_rad = self.rotation[2]
            
            # Major axis endpoint
            axis_x = center_2d[0] + self.semi_axes[0] * np.cos(angle_rad)
            axis_y = center_2d[1] + self.semi_axes[0] * np.sin(angle_rad)
            
            # Draw line from center to axis endpoint
            ax.plot([center_2d[0], axis_x], [center_2d[1], axis_y], 
                   color=color, linewidth=linewidth, linestyle='-', alpha=alpha)
            
            # Draw axis endpoint marker (square)
            ax.plot(axis_x, axis_y, 's', 
                   color=color, markersize=10, markerfacecolor=color)
    
    def get_control_points(self) -> List[ControlPoint]:
        """
        Get draggable control points for interactive UI.
        
        Returns:
            - Point 0: Center (x, y)
            - Point 1: Major axis endpoint (for rotation and scaling)
        """
        center_2d = self.center[:2]
        angle_rad = self.rotation[2]
        
        # Major axis endpoint
        axis_x = center_2d[0] + self.semi_axes[0] * np.cos(angle_rad)
        axis_y = center_2d[1] + self.semi_axes[0] * np.sin(angle_rad)
        
        return [
            ControlPoint(
                x=float(center_2d[0]),
                y=float(center_2d[1]),
                point_type='center',
                index=0
            ),
            ControlPoint(
                x=float(axis_x),
                y=float(axis_y),
                point_type='axis',
                index=1
            )
        ]
    
    def update_from_control_point(self, point_index: int, new_x: float, new_y: float):
        """
        Update ellipsoid when a control point is dragged.
        
        Args:
            point_index: 0 for center, 1 for axis endpoint
            new_x, new_y: New position
        """
        if point_index == 0:
            # Moving center
            self.center[0] = new_x
            self.center[1] = new_y
        
        elif point_index == 1:
            # Moving axis endpoint: updates both rotation and size
            center_2d = self.center[:2]
            
            # Calculate new axis vector
            axis_vec = np.array([new_x - center_2d[0], new_y - center_2d[1]])
            
            # Calculate new length (semi-axis a)
            new_length = np.linalg.norm(axis_vec)
            if new_length > 1e-6:  # Avoid division by zero
                self.semi_axes[0] = new_length
                
                # Calculate new rotation angle
                new_angle = np.arctan2(axis_vec[1], axis_vec[0])
                self.rotation[2] = new_angle
        
        else:
            raise ValueError(f"Invalid control point index: {point_index}")
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {
            'center': self.center.copy(),
            'semi_axes': self.semi_axes.copy(),
            'rotation': self.rotation.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        self.center = np.array(params['center'], dtype=np.float32)
        self.semi_axes = np.array(params['semi_axes'], dtype=np.float32)
        self.rotation = np.array(params['rotation'], dtype=np.float32)
    
    def get_bounding_box(self) -> tuple:
        """
        Get 2D bounding box of the ellipse.
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        # For a rotated ellipse, the bounding box is not trivial
        # We'll sample points on the ellipse and find min/max
        
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Points on unit circle
        unit_x = np.cos(theta)
        unit_y = np.sin(theta)
        
        # Scale by semi-axes
        scaled_x = self.semi_axes[0] * unit_x
        scaled_y = self.semi_axes[1] * unit_y
        
        # Rotate by yaw angle
        angle = self.rotation[2]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotated_x = cos_a * scaled_x - sin_a * scaled_y
        rotated_y = sin_a * scaled_x + cos_a * scaled_y
        
        # Translate to center
        final_x = rotated_x + self.center[0]
        final_y = rotated_y + self.center[1]
        
        return (
            float(np.min(final_x)),
            float(np.min(final_y)),
            float(np.max(final_x)),
            float(np.max(final_y))
        )
    
    def get_area(self) -> float:
        """Calculate the area of the ellipse (2D projection)."""
        return float(np.pi * self.semi_axes[0] * self.semi_axes[1])
    
    def get_volume(self) -> float:
        """Calculate the volume of the 3D ellipsoid."""
        return float((4.0 / 3.0) * np.pi * np.prod(self.semi_axes))
    
    def contains_point(self, x: float, y: float, z: float = 0.0) -> bool:
        """
        Check if a 3D point is inside the ellipsoid.
        
        Args:
            x, y, z: Point coordinates
            
        Returns:
            True if point is inside ellipsoid
        """
        # Transform point to ellipsoid local coordinates
        point = np.array([x, y, z])
        
        # Translate to origin
        translated = point - self.center
        
        # Rotate to align with axes (inverse rotation)
        # For simplicity, we'll just check against unrotated ellipsoid
        # Full implementation would require rotation matrix
        
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
        normalized = translated / self.semi_axes
        distance_squared = np.sum(normalized ** 2)
        
        return distance_squared <= 1.0
    
    def __repr__(self) -> str:
        return (f"EllipsoidModel(frame_idx={self.frame_idx}, "
                f"center={self.center}, semi_axes={self.semi_axes}, "
                f"rotation={self.rotation})")


def create_ellipsoid_params(frame_idx: int, center: List[float], 
                           semi_axes: List[float], rotation: List[float]) -> EllipsoidModel:
    """
    Convenience function to create ellipsoid parameters.
    
    Args:
        frame_idx: Frame index
        center: [x, y, z] center position
        semi_axes: [a, b, c] semi-axis lengths
        rotation: [roll, pitch, yaw] rotation angles in radians
        
    Returns:
        EllipsoidModel instance
    """
    return EllipsoidModel(
        frame_idx=frame_idx,
        center=np.array(center),
        semi_axes=np.array(semi_axes),
        rotation=np.array(rotation)
    )
