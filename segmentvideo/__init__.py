"""
SegmentVideo: Assisted Manual Segmentation for Video Feature Tracking

A Python framework for easing manual video annotation by leveraging 
temporal correlation and intelligent interpolation.
"""

__version__ = "0.1.0"
__author__ = "kularos"

from segmentvideo.models.base import BaseModel
from segmentvideo.models.ellipsoid import EllipsoidModel
from segmentvideo.models.curve import CurveModel
from segmentvideo.annotation.state import AnnotationState
from segmentvideo.io.video_loader import load_video_to_tensor, get_video_info

__all__ = [
    "BaseModel",
    "EllipsoidModel", 
    "CurveModel",
    "AnnotationState",
    "load_video_to_tensor",
    "get_video_info",
]
