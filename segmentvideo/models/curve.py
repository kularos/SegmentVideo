"""
Curve/Chain model with B-spline and interpolating spline support.

Uses B-spline for optimal representation and analysis.
Converts to interpolating spline for user editing.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.integrate import cumulative_trapezoid
from .base import BaseModel, ControlPoint


class CurveModel(BaseModel):
    """
    Spline-based curve model with support for curvature analysis.
    
    Internal representation: B-spline (smooth, optimal for analysis)
    Editing mode: Interpolating spline (intuitive for user interaction)
    """
    
    def __init__(self, frame_idx: int, points: np.ndarray, use_bspline: bool = True, 
                 smoothing: float = 0.0, anchor_tangent: Optional[np.ndarray] = None):
        """
        Initialize curve model.
        
        Args:
            frame_idx: Frame index
            points: Numpy array of shape (N, 2) with control/interpolation points
            use_bspline: If True, fit B-spline. If False, use interpolating cubic spline
            smoothing: Smoothing factor for B-spline (0 = interpolating, >0 = approximating)
            anchor_tangent: Optional (dx, dy) unit vector for tangent constraint at first point
        """
        super().__init__(frame_idx)
        self.points = np.array(points, dtype=np.float32)
        self.n_points = len(self.points)
        self.use_bspline = use_bspline
        self.smoothing = smoothing
        self.anchor_tangent = anchor_tangent
        
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Points must be (N, 2), got {self.points.shape}")
        
        # Fit spline
        self._fit_spline()
    
    def _fit_spline(self):
        """Fit spline to control points."""
        if self.use_bspline:
            self._fit_bspline()
        else:
            self._fit_interpolating_spline()
    
    def _fit_bspline(self):
        """Fit B-spline to control points."""
        # Use splprep for parametric B-spline
        # s=smoothing: 0=interpolating, >0=smoothing (reduces oscillations)
        try:
            self.tck, self.u = splprep(
                [self.points[:, 0], self.points[:, 1]], 
                s=self.smoothing,  # Smoothing factor
                k=3,  # Cubic spline
                per=False  # Not periodic
            )
            self.spline_type = 'bspline'
        except Exception as e:
            print(f"Warning: B-spline fitting failed ({e}), falling back to interpolating spline")
            self._fit_interpolating_spline()
    
    def _fit_interpolating_spline(self):
        """Fit interpolating cubic spline to control points."""
        # Parameterize by cumulative chord length
        distances = np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
        self.u = cumulative_dist / cumulative_dist[-1] if cumulative_dist[-1] > 0 else np.linspace(0, 1, len(self.points))
        
        # Determine boundary conditions
        if self.anchor_tangent is not None:
            # Enforce tangent at first point (anchor)
            # bc_type: ((1st deriv at start), (1st deriv at end)) or 'natural' or 'clamped'
            # For now, use 'clamped' (zero second derivative at boundaries)
            bc_type = 'clamped'
        else:
            bc_type = 'natural'
        
        # Fit separate splines for x and y
        self.cs_x = CubicSpline(self.u, self.points[:, 0], bc_type=bc_type)
        self.cs_y = CubicSpline(self.u, self.points[:, 1], bc_type=bc_type)
        self.spline_type = 'interpolating'
    
    @classmethod
    def from_contour(cls, frame_idx: int, contour_points: np.ndarray, n_points: int,
                     use_bspline: bool = True, smoothing: float = 0.0,
                     anchor_tangent: Optional[np.ndarray] = None) -> 'CurveModel':
        """
        Create a curve model from a contour by resampling to N evenly-spaced points.
        
        Args:
            frame_idx: Frame index
            contour_points: Numpy array of shape (M, 2) with contour coordinates
            n_points: Number of control points to use
            use_bspline: If True, use B-spline representation
            smoothing: Smoothing factor for B-spline (0 = passes through all points)
            anchor_tangent: Optional (dx, dy) unit vector for tangent at first point
            
        Returns:
            CurveModel instance
        """
        # Calculate cumulative arc length along contour
        diffs = np.diff(contour_points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length == 0:
            raise ValueError("Contour has zero length")
        
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
        
        return cls(frame_idx, resampled_points, use_bspline=use_bspline, 
                  smoothing=smoothing, anchor_tangent=anchor_tangent)
    
    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate spline at parameter values t.
        
        Args:
            t: Parameter values in [0, 1]
            
        Returns:
            Array of shape (len(t), 2) with (x, y) coordinates
        """
        t = np.clip(t, 0, 1)  # Ensure t is in valid range
        
        if self.spline_type == 'bspline':
            # Evaluate B-spline
            x, y = splev(t, self.tck)
            return np.column_stack([x, y])
        else:
            # Evaluate interpolating spline
            x = self.cs_x(t)
            y = self.cs_y(t)
            return np.column_stack([x, y])
    
    def compute_curvature(self, t: np.ndarray) -> np.ndarray:
        """
        Compute curvature at parameter values t.
        
        Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        
        Args:
            t: Parameter values in [0, 1]
            
        Returns:
            Array of curvature values
        """
        t = np.clip(t, 0, 1)
        
        if self.spline_type == 'bspline':
            # First derivatives
            dx, dy = splev(t, self.tck, der=1)
            # Second derivatives
            ddx, ddy = splev(t, self.tck, der=2)
        else:
            # First derivatives
            dx = self.cs_x(t, 1)
            dy = self.cs_y(t, 1)
            # Second derivatives
            ddx = self.cs_x(t, 2)
            ddy = self.cs_y(t, 2)
        
        # Curvature formula
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        
        # Avoid division by zero
        curvature = np.where(denominator > 1e-10, numerator / denominator, 0)
        
        return curvature
    
    def compute_arc_length_parameterization(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute arc-length parameterization s(t).
        
        Args:
            n_samples: Number of samples for integration
            
        Returns:
            (t_values, s_values): Parameter and corresponding arc-length
        """
        t_values = np.linspace(0, 1, n_samples)
        points = self.evaluate(t_values)
        
        # Compute arc length by integrating ||dr/dt||
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt((diffs**2).sum(axis=1))
        s_values = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        return t_values, s_values
    
    def evaluate_at_arc_length(self, s: np.ndarray) -> np.ndarray:
        """
        Evaluate spline at arc-length values s.
        
        Args:
            s: Arc-length values
            
        Returns:
            Array of (x, y) coordinates
        """
        # Compute arc-length parameterization
        t_values, s_values = self.compute_arc_length_parameterization()
        
        # Interpolate to find t(s)
        t_at_s = np.interp(s, s_values, t_values)
        
        # Evaluate at these t values
        return self.evaluate(t_at_s)
    
    def get_total_length(self) -> float:
        """Calculate total arc length of the curve."""
        _, s_values = self.compute_arc_length_parameterization()
        return float(s_values[-1])
    
    def get_backbone_length(self) -> float:
        """Calculate distance between endpoints (straight-line distance)."""
        return float(np.linalg.norm(self.points[-1] - self.points[0]))
    
    def to_interpolating_for_editing(self, n_edit_points: Optional[int] = None) -> 'CurveModel':
        """
        Convert current spline to interpolating spline for editing.
        
        If already interpolating, returns self.
        If B-spline, samples points and creates interpolating version.
        
        Args:
            n_edit_points: Number of edit points to extract (default: same as control points)
            
        Returns:
            CurveModel with interpolating spline
        """
        if self.spline_type == 'interpolating':
            return self
        
        # Extract evenly-spaced points from B-spline
        n_edit = n_edit_points if n_edit_points is not None else self.n_points
        t_edit = np.linspace(0, 1, n_edit)
        edit_points = self.evaluate(t_edit)
        
        # Create interpolating spline
        return CurveModel(self.frame_idx, edit_points, use_bspline=False, 
                         anchor_tangent=self.anchor_tangent)
    
    def to_bspline_from_editing(self, smoothing: float = 0.05) -> 'CurveModel':
        """
        Convert interpolating spline back to B-spline after editing.
        
        Args:
            smoothing: Smoothing factor for B-spline fit
            
        Returns:
            CurveModel with B-spline
        """
        return CurveModel(self.frame_idx, self.points, use_bspline=True, 
                         smoothing=smoothing, anchor_tangent=self.anchor_tangent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'model_type': 'CurveModel',
            'frame_idx': self.frame_idx,
            'points': self.points.tolist(),
            'n_points': self.n_points,
            'use_bspline': self.use_bspline,
            'smoothing': self.smoothing,
            'spline_type': self.spline_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurveModel':
        """Deserialize from dictionary."""
        return cls(
            frame_idx=data['frame_idx'],
            points=np.array(data['points']),
            use_bspline=data.get('use_bspline', True),
            smoothing=data.get('smoothing', 0.0)
        )
    
    def interpolate(self, other: 'CurveModel', alpha: float) -> 'CurveModel':
        """
        Interpolate between this curve and another.
        
        Both curves must have the same number of control points.
        """
        if not isinstance(other, CurveModel):
            raise TypeError(f"Cannot interpolate CurveModel with {type(other)}")
        
        if self.n_points != other.n_points:
            raise ValueError(f"Cannot interpolate curves with different point counts: {self.n_points} vs {other.n_points}")
        
        # Linear interpolation of control points
        interpolated_points = (1 - alpha) * self.points + alpha * other.points
        
        # Calculate interpolated frame index
        interpolated_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)
        
        # Use same spline settings as self
        return CurveModel(interpolated_frame_idx, interpolated_points, 
                         use_bspline=self.use_bspline, smoothing=self.smoothing)
    
    def render(self, ax, color='red', linewidth=2, show_points=True, 
               show_backbone=True, n_eval: int = 200, **kwargs):
        """
        Render the curve on matplotlib axes.
        
        Args:
            ax: Matplotlib axes
            color: Color for the curve
            linewidth: Line width
            show_points: Whether to show control points
            show_backbone: Whether to show the dashed backbone between endpoints
            n_eval: Number of points to evaluate for smooth display
        """
        # Evaluate spline at many points for smooth display
        t_eval = np.linspace(0, 1, n_eval)
        curve_points = self.evaluate(t_eval)
        
        # Draw the smooth curve
        ax.plot(curve_points[:, 0], curve_points[:, 1], 
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
            
            # Draw middle control points as circles
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
        
        For spline-based curves, simply update the control point and re-fit.
        """
        if point_index < 0 or point_index >= self.n_points:
            raise ValueError(f"Invalid point index: {point_index}")
        
        # Update control point
        self.points[point_index] = np.array([new_x, new_y])
        
        # Re-fit spline
        self._fit_spline()
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {
            'points': self.points.copy(),
            'use_bspline': self.use_bspline,
            'smoothing': self.smoothing
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        self.points = np.array(params['points'], dtype=np.float32)
        self.n_points = len(self.points)
        self.use_bspline = params.get('use_bspline', True)
        self.smoothing = params.get('smoothing', 0.0)
        self._fit_spline()
