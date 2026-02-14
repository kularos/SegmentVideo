"""
Base model class for all primitive tracking models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ControlPoint:
    """Represents a draggable control point in the UI."""
    x: float
    y: float
    point_type: str  # 'endpoint', 'middle', 'center', 'axis'
    index: int  # Index in the model's parameter array


class BaseModel(ABC):
    """Abstract base class for all tracking models (ellipsoid, curve, etc.)."""
    
    def __init__(self, frame_idx: int):
        self.frame_idx = frame_idx
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Deserialize model from dictionary."""
        pass
    
    @abstractmethod
    def interpolate(self, other: 'BaseModel', alpha: float) -> 'BaseModel':
        """
        Interpolate between this model and another.
        
        Args:
            other: Another model of the same type
            alpha: Interpolation factor (0.0 = this model, 1.0 = other model)
            
        Returns:
            New model instance with interpolated parameters
        """
        pass
    
    @abstractmethod
    def render(self, ax, **kwargs):
        """
        Render the model on a matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
            **kwargs: Additional rendering options (color, linewidth, etc.)
        """
        pass
    
    @abstractmethod
    def get_control_points(self) -> List[ControlPoint]:
        """
        Get draggable control points for interactive UI.
        
        Returns:
            List of ControlPoint objects
        """
        pass
    
    @abstractmethod
    def update_from_control_point(self, point_index: int, new_x: float, new_y: float):
        """
        Update model parameters when a control point is dragged.
        
        Args:
            point_index: Index of the control point being dragged
            new_x: New x coordinate
            new_y: New y coordinate
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters as a dictionary of numpy arrays."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters from a dictionary of numpy arrays."""
        pass
    
    def copy(self) -> 'BaseModel':
        """Create a deep copy of this model."""
        return self.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        return f"{self.model_type}(frame_idx={self.frame_idx})"
