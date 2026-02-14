"""
Watershed segmentation for initial feature detection.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class WatershedSeed:
    """Represents a user-placed seed point for watershed segmentation."""
    x: float
    y: float
    marker_id: int  # 1 = background, 2+ = features
    color: str


@dataclass
class WatershedFeature:
    """Represents a segmented feature from watershed."""
    marker_id: int
    mask: np.ndarray  # Binary mask of the feature
    contours: List[np.ndarray]  # List of contour arrays
    color: str


class WatershedSegmenter:
    """
    Handles watershed segmentation for initial feature detection.
    """
    
    FEATURE_COLORS = ['deepskyblue', 'lime', 'magenta', 'yellow', 'orange', 'cyan', 'pink']
    
    def __init__(self, image: np.ndarray):
        """
        Initialize watershed segmenter.
        
        Args:
            image: Input image as (H, W, 3) RGB numpy array
        """
        self.image_rgb = image
        self.H, self.W = image.shape[:2]
        
        # Convert to BGR for OpenCV
        self.image_bgr = cv2.cvtColor(
            (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        
        self.seeds: List[WatershedSeed] = []
        self.markers: Optional[np.ndarray] = None
        self.features: Dict[int, WatershedFeature] = {}
    
    def add_seed(self, x: float, y: float, marker_id: int) -> WatershedSeed:
        """
        Add a seed point for watershed.
        
        Args:
            x, y: Pixel coordinates
            marker_id: 1 for background, 2+ for features
            
        Returns:
            WatershedSeed object
        """
        color = self.FEATURE_COLORS[(marker_id - 1) % len(self.FEATURE_COLORS)]
        seed = WatershedSeed(x, y, marker_id, color)
        self.seeds.append(seed)
        return seed
    
    def remove_seed(self, index: int):
        """Remove a seed by index."""
        if 0 <= index < len(self.seeds):
            self.seeds.pop(index)
    
    def clear_seeds(self):
        """Clear all seeds."""
        self.seeds.clear()
    
    def run_watershed(self, blur_kernel: int = 5) -> np.ndarray:
        """
        Run watershed segmentation based on current seeds.
        
        Args:
            blur_kernel: Size of Gaussian blur kernel (must be odd)
            
        Returns:
            Markers array where each pixel is labeled with its marker_id
        """
        if not self.seeds:
            raise ValueError("No seeds placed. Add seeds before running watershed.")
        
        # Initialize markers
        markers = np.zeros((self.H, self.W), dtype=np.int32)
        
        # Set borders to background (marker_id = 1)
        markers[0, :] = 1
        markers[-1, :] = 1
        markers[:, 0] = 1
        markers[:, -1] = 1
        
        # Place seed markers
        for seed in self.seeds:
            cv2.circle(markers, (int(seed.x), int(seed.y)), 3, seed.marker_id, -1)
        
        # Apply Gaussian blur to image
        blurred = cv2.GaussianBlur(self.image_bgr, (blur_kernel, blur_kernel), 0)
        
        # Run watershed
        self.markers = cv2.watershed(blurred, markers)
        
        # Extract features
        self._extract_features()
        
        return self.markers
    
    def _extract_features(self):
        """Extract feature information from watershed result."""
        if self.markers is None:
            return
        
        self.features.clear()
        
        # Find all unique marker IDs (excluding -1 for boundaries and 1 for background)
        unique_markers = np.unique(self.markers)
        feature_markers = [m for m in unique_markers if m > 1]
        
        for marker_id in feature_markers:
            # Create binary mask
            mask = (self.markers == marker_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Get color
            color = self.FEATURE_COLORS[marker_id % len(self.FEATURE_COLORS)]
            
            self.features[marker_id] = WatershedFeature(
                marker_id=marker_id,
                mask=mask,
                contours=contours,
                color=color
            )
    
    def get_feature_edge_contour(self, marker_id: int, edge_point: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Extract the edge contour of a feature closest to a specified edge point.
        
        This is used when the user clicks on a point to indicate which edge
        of the feature they want to track.
        
        Args:
            marker_id: Feature marker ID
            edge_point: (x, y) coordinates of user-clicked edge point
            
        Returns:
            Numpy array of shape (N, 2) with contour points, or None if feature not found
        """
        if marker_id not in self.features:
            return None
        
        feature = self.features[marker_id]
        
        if not feature.contours:
            return None
        
        # Find the largest contour (main feature outline)
        main_contour = max(feature.contours, key=cv2.contourArea)
        
        # Reshape contour from (N, 1, 2) to (N, 2)
        contour_points = main_contour.reshape(-1, 2).astype(np.float32)
        
        # Find the point on the contour closest to the edge_point
        edge_array = np.array(edge_point, dtype=np.float32)
        distances = np.linalg.norm(contour_points - edge_array, axis=1)
        closest_idx = np.argmin(distances)
        
        # For now, return the full contour
        # In a more advanced implementation, we could extract just one side
        # of the contour relative to the clicked point
        
        return contour_points
    
    def extract_feature_skeleton(self, marker_id: int) -> Optional[np.ndarray]:
        """
        Extract the skeleton (medial axis) of a feature.
        
        This is useful for thin/elongated features where the skeleton
        represents the centerline.
        
        Args:
            marker_id: Feature marker ID
            
        Returns:
            Numpy array of shape (N, 2) with skeleton points, or None
        """
        if marker_id not in self.features:
            return None
        
        feature = self.features[marker_id]
        
        # Skeletonization using morphological operations
        skeleton = np.zeros(feature.mask.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        temp = feature.mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = eroded - opened
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        # Extract skeleton points
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) == 0:
            return None
        
        # Convert from (row, col) to (x, y)
        skeleton_points = skeleton_points[:, [1, 0]].astype(np.float32)
        
        return skeleton_points
    
    def get_overlay(self, alpha: float = 0.4) -> np.ndarray:
        """
        Create colored overlay showing segmented features.
        
        Args:
            alpha: Transparency (0-1)
            
        Returns:
            RGBA overlay array of shape (H, W, 4)
        """
        if self.markers is None:
            return np.zeros((self.H, self.W, 4))
        
        from matplotlib.colors import to_rgba
        
        overlay = np.zeros((self.H, self.W, 4))
        
        for marker_id, feature in self.features.items():
            color = to_rgba(feature.color)
            overlay[self.markers == marker_id] = [*color[:3], alpha]
        
        return overlay
    
    def get_feature_info(self, marker_id: int) -> Optional[Dict]:
        """
        Get information about a specific feature.
        
        Returns:
            Dictionary with feature properties or None
        """
        if marker_id not in self.features:
            return None
        
        feature = self.features[marker_id]
        
        # Calculate properties
        area = np.sum(feature.mask)
        
        if feature.contours:
            main_contour = max(feature.contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, closed=True)
            
            # Fit bounding rectangle
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            
            # Calculate centroid
            M = cv2.moments(main_contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = rect[0]
        else:
            perimeter = 0
            box = None
            cx, cy = 0, 0
        
        return {
            'marker_id': marker_id,
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'bounding_box': box,
            'color': feature.color,
            'n_contours': len(feature.contours)
        }
