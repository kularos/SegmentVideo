"""
Curve/Chain model with evenly-spaced control points.
"""

from typing import List, Dict, Any
import numpy as np
from .base import BaseModel, ControlPoint


class CurveModel(BaseModel):
    """
    Chain/curve model with N evenly-spaced control points.
    
    Implements constant spacing constraint where:
    - Endpoints can move freely
    - Middle points can only move perpendicular to the backbone
    - All points maintain equal spacing along the chain
    """
    
    def __init__(self, frame_idx: int, points: np.ndarray):
        """
        Initialize curve model.
        
        Args:
            frame_idx: Frame index
            points: Numpy array of shape (N, 2) or (N, 3) with control point coordinates
                   Format: [[x1, y1], [x2, y2], ..., [xN, yN]]
                   or [[x1, y1, z1], [x2, y2, z2], ..., [xN, yN, zN]]
        """
        super().__init__(frame_idx)
        self.points = np.array(points, dtype=np.float32)
        self.n_points = len(self.points)
        
        if self.points.ndim != 2 or self.points.shape[1] not in [2, 3]:
            raise ValueError(f"Points must be (N, 2) or (N, 3), got {self.points.shape}")
    
    @classmethod
    def from_contour(cls, frame_idx: int, contour_points: np.ndarray, n_points: int) -> 'CurveModel':
        """
        Create a curve model from a contour by resampling to N evenly-spaced points.
        
        Args:
            frame_idx: Frame index
            contour_points: Numpy array of shape (M, 2) with contour coordinates
            n_points: Number of control points to use for the chain model
            
        Returns:
            CurveModel instance
        """
        # Calculate cumulative arc length along contour
        diffs = np.diff(contour_points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        
        # Resample at evenly-spaced arc lengths
        target_lengths = np.linspace(0, total_length, n_points)
        resampled_points = np.zeros((n_points, 2))
        
        for i, target_len in enumerate(target_lengths):
            # Find the segment containing this arc length
            idx = np.searchsorted(cumulative_length, target_len)
            if idx == 0:
                resampled_points[i] = contour_points[0]
            elif idx >= len(contour_points):
                resampled_points[i] = contour_points[-1]
            else:
                # Interpolate within the segment
                segment_start_len = cumulative_length[idx - 1]
                segment_end_len = cumulative_length[idx]
                alpha = (target_len - segment_start_len) / (segment_end_len - segment_start_len)
                resampled_points[i] = (1 - alpha) * contour_points[idx - 1] + alpha * contour_points[idx]
        
        return cls(frame_idx, resampled_points)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'model_type': 'CurveModel',
            'frame_idx': self.frame_idx,
            'points': self.points.tolist(),
            'n_points': self.n_points
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurveModel':
        """Deserialize from dictionary."""
        return cls(
            frame_idx=data['frame_idx'],
            points=np.array(data['points'])
        )
    
    def interpolate(self, other: 'CurveModel', alpha: float) -> 'CurveModel':
        """
        Interpolate between this curve and another.
        
        Both curves must have the same number of points.
        """
        if not isinstance(other, CurveModel):
            raise TypeError(f"Cannot interpolate CurveModel with {type(other)}")
        
        if self.n_points != other.n_points:
            raise ValueError(f"Cannot interpolate curves with different point counts: {self.n_points} vs {other.n_points}")
        
        # Linear interpolation of all points
        interpolated_points = (1 - alpha) * self.points + alpha * other.points
        
        # Calculate interpolated frame index
        interpolated_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)
        
        return CurveModel(interpolated_frame_idx, interpolated_points)
    
    def render(self, ax, color='red', linewidth=2, show_points=True, 
               show_backbone=True, **kwargs):
        """
        Render the curve on matplotlib axes.
        
        Args:
            ax: Matplotlib axes
            color: Color for the curve
            linewidth: Line width
            show_points: Whether to show control points
            show_backbone: Whether to show the dashed backbone between endpoints
        """
        # Draw the curve through all points
        ax.plot(self.points[:, 0], self.points[:, 1], 
                color=color, linewidth=linewidth, **kwargs)
        
        if show_backbone and self.n_points > 2:
            # Draw dashed line between endpoints
            ax.plot([self.points[0, 0], self.points[-1, 0]], 
                   [self.points[0, 1], self.points[-1, 1]], 
                   color=color, linestyle='--', linewidth=linewidth * 0.5, alpha=0.5)
        
        if show_points:
            # Draw endpoints as squares
            ax.plot([self.points[0, 0], self.points[-1, 0]], 
                   [self.points[0, 1], self.points[-1, 1]], 
                   'rs', markersize=10, markerfacecolor=color)
            
            # Draw middle points as circles
            if self.n_points > 2:
                ax.plot(self.points[1:-1, 0], self.points[1:-1, 1], 
                       'ro', markersize=8, markerfacecolor=color)
    
    def get_control_points(self) -> List[ControlPoint]:
        """Get all points as draggable control points."""
        control_points = []
        
        for i, point in enumerate(self.points):
            if i == 0 or i == self.n_points - 1:
                point_type = 'endpoint'
            else:
                point_type = 'middle'
            
            control_points.append(ControlPoint(
                x=float(point[0]),
                y=float(point[1]),
                point_type=point_type,
                index=i
            ))
        
        return control_points
    
    def update_from_control_point(self, point_index: int, new_x: float, new_y: float):
        """
        Update curve when a control point is dragged.
        
        Implements constant spacing constraint:
        - Endpoints: Can move freely, redistributes all points
        - Middle points: Can only move perpendicular to backbone
        """
        if point_index < 0 or point_index >= self.n_points:
            raise ValueError(f"Invalid point index: {point_index}")
        
        if point_index == 0 or point_index == self.n_points - 1:
            # Endpoint moved: update and redistribute all points
            self._update_endpoint(point_index, new_x, new_y)
        else:
            # Middle point moved: constrain to perpendicular motion
            self._update_middle_point(point_index, new_x, new_y)
    
    def _update_endpoint(self, endpoint_idx: int, new_x: float, new_y: float):
        """
        Update an endpoint and redistribute all points with constant spacing.
        
        Uses complex number rotation to preserve curvature shape.
        """
        old_endpoint = self.points[endpoint_idx].copy()
        
        # Determine which endpoint moved
        if endpoint_idx == 0:
            fixed_endpoint_idx = self.n_points - 1
        else:
            fixed_endpoint_idx = 0
        
        fixed_endpoint = self.points[fixed_endpoint_idx]
        
        # Calculate old and new backbone vectors
        old_backbone = self.points[fixed_endpoint_idx] - old_endpoint
        new_backbone = fixed_endpoint - np.array([new_x, new_y])
        
        # Calculate rotation angle using complex numbers
        # z = x + iy
        old_z = complex(old_backbone[0], old_backbone[1])
        new_z = complex(new_backbone[0], new_backbone[1])
        
        if abs(old_z) < 1e-6:
            # Degenerate case: old backbone has zero length
            rotation = complex(1, 0)
        else:
            rotation = new_z / old_z
        
        # Rotate perpendicular offsets for all middle points
        for i in range(1, self.n_points - 1):
            # Calculate old offset from backbone
            t = i / (self.n_points - 1)  # Parameter along backbone [0, 1]
            old_backbone_point = old_endpoint + t * old_backbone
            old_offset = self.points[i] - old_backbone_point
            
            # Rotate the offset
            offset_z = complex(old_offset[0], old_offset[1])
            rotated_offset_z = offset_z * rotation
            rotated_offset = np.array([rotated_offset_z.real, rotated_offset_z.imag])
            
            # New position on new backbone + rotated offset
            new_backbone_point = np.array([new_x, new_y]) + t * new_backbone
            self.points[i] = new_backbone_point + rotated_offset
        
        # Update the moved endpoint
        self.points[endpoint_idx] = np.array([new_x, new_y])
    
    def _update_middle_point(self, point_idx: int, new_x: float, new_y: float):
        """
        Update a middle point with perpendicular constraint.
        
        Projects the new position onto a line perpendicular to the backbone.
        """
        # Get backbone (line between endpoints)
        p_start = self.points[0]
        p_end = self.points[-1]
        backbone = p_end - p_start
        backbone_length = np.linalg.norm(backbone)
        
        if backbone_length < 1e-6:
            # Degenerate case: endpoints are at the same location
            return
        
        backbone_unit = backbone / backbone_length
        
        # Calculate where this point should be on the backbone
        t = point_idx / (self.n_points - 1)
        backbone_point = p_start + t * backbone
        
        # Calculate new offset vector
        new_offset = np.array([new_x, new_y]) - backbone_point
        
        # Project offset onto perpendicular direction
        # Perpendicular to backbone: rotate backbone by 90 degrees
        perpendicular = np.array([-backbone_unit[1], backbone_unit[0]])
        projected_offset_magnitude = np.dot(new_offset, perpendicular)
        projected_offset = projected_offset_magnitude * perpendicular
        
        # Update point
        self.points[point_idx] = backbone_point + projected_offset
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {
            'points': self.points.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        self.points = np.array(params['points'], dtype=np.float32)
        self.n_points = len(self.points)
    
    def get_total_length(self) -> float:
        """Calculate total arc length of the curve."""
        diffs = np.diff(self.points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        return float(segment_lengths.sum())
    
    def get_backbone_length(self) -> float:
        """Calculate distance between endpoints (backbone length)."""
        return float(np.linalg.norm(self.points[-1] - self.points[0]))
