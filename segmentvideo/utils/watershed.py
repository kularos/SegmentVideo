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
        if marker_id == 1:
            color = 'lightblue'  # Background
        else:
            # Feature 1 (marker_id=2) -> FEATURE_COLORS[1] (lime)
            # Feature 2 (marker_id=3) -> FEATURE_COLORS[2] (magenta)
            color = self.FEATURE_COLORS[marker_id - 1]
        
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
        
        # Place seed markers directly
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
            
            # Get color: marker_id 2 -> FEATURE_COLORS[1] (lime), marker_id 3 -> FEATURE_COLORS[2] (magenta)
            color = self.FEATURE_COLORS[marker_id - 1]
            
            self.features[marker_id] = WatershedFeature(
                marker_id=marker_id,
                mask=mask,
                contours=contours,
                color=color
            )
    
    def get_feature_edge_contour_with_anchor(self, marker_id: int, edge_point: Tuple[float, float],
                                             base_anchor: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Extract the edge contour of a feature on the side indicated by edge_point.
        
        The contour starts at base_anchor and ends at the tip (furthest point from base).
        
        Strategy:
        1. Find closest contour point to base_anchor (this is the base)
        2. Find the tip point (furthest from base)
        3. Split contour at base and tip into two sides
        4. Choose the side closest to edge_point
        5. Return ordered points from base to tip
        
        Args:
            marker_id: Feature marker ID
            edge_point: (x, y) coordinates indicating which side to track
            base_anchor: (x, y) coordinates of the base anchor point (from box fitting)
            
        Returns:
            Numpy array of shape (N, 2) with contour points from base to tip
        """
        if marker_id not in self.features:
            return None
        
        feature = self.features[marker_id]
        
        if not feature.contours:
            return None
        
        # Find the largest contour (main feature outline)
        main_contour = max(feature.contours, key=cv2.contourArea)
        contour_points = main_contour.reshape(-1, 2).astype(np.float32)
        
        # Step 1: Find base point (closest contour point to base_anchor)
        anchor_array = np.array(base_anchor, dtype=np.float32)
        distances_to_anchor = np.linalg.norm(contour_points - anchor_array, axis=1)
        base_idx = np.argmin(distances_to_anchor)
        base_point = contour_points[base_idx]
        
        # Step 2: Find tip point (furthest from base)
        distances_from_base = np.linalg.norm(contour_points - base_point, axis=1)
        tip_idx = np.argmax(distances_from_base)
        tip_point = contour_points[tip_idx]
        
        # Step 3: Split contour into two sides
        # Contour is a closed loop, so we need to split it at base and tip
        n = len(contour_points)
        
        if base_idx < tip_idx:
            # Side 1: base -> tip (going forward)
            side1 = contour_points[base_idx:tip_idx+1]
            # Side 2: tip -> base (going around)
            side2 = np.vstack([contour_points[tip_idx:], contour_points[:base_idx+1]])
        else:
            # Side 1: base -> tip (going around)
            side1 = np.vstack([contour_points[base_idx:], contour_points[:tip_idx+1]])
            # Side 2: tip -> base (going forward)
            side2 = contour_points[tip_idx:base_idx+1]
        
        # Step 4: Choose side closest to edge_point
        edge_array = np.array(edge_point, dtype=np.float32)
        
        # Calculate average distance from each side to edge_point
        dist1 = np.mean(np.linalg.norm(side1 - edge_array, axis=1))
        dist2 = np.mean(np.linalg.norm(side2 - edge_array, axis=1))
        
        # Choose the closer side
        if dist1 < dist2:
            selected_side = side1
            print(f"Selected side 1 (avg dist: {dist1:.1f} px)")
        else:
            selected_side = side2[::-1]  # Reverse to go base->tip
            print(f"Selected side 2 (avg dist: {dist2:.1f} px)")
        
        print(f"Contour extracted: {len(selected_side)} points from base to tip")
        print(f"  Base: ({selected_side[0][0]:.1f}, {selected_side[0][1]:.1f})")
        print(f"  Tip: ({selected_side[-1][0]:.1f}, {selected_side[-1][1]:.1f})")
        
        return selected_side
    
    def get_feature_edge_contour(self, marker_id: int, edge_point: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Extract the edge contour of a feature closest to a specified edge point.
        
        This is the legacy method that doesn't use an anchor.
        For new code, use get_feature_edge_contour_with_anchor instead.
        
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
            # Skip background (marker_id=1)
            if marker_id == 1:
                continue
            
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
