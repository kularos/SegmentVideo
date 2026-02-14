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
    
    def select_edge_by_gradient(self, marker_id: int, base_anchor: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Automatically select front vs back edge using gradient analysis.
        
        For a 3D strip viewed in perspective, the front edge has a sharper gradient
        (strip→background transition) than the back edge.
        
        Strategy:
        1. Extract both edges of the feature (split contour at base and tip)
        2. For each edge, compute gradient magnitude perpendicular to edge
        3. Select edge with stronger gradient (front edge)
        
        Args:
            marker_id: Feature marker ID
            base_anchor: Base anchor point (x, y)
            
        Returns:
            Contour points of the front edge (stronger gradient), ordered base→tip
        """
        if marker_id not in self.features:
            return None
        
        feature = self.features[marker_id]
        
        if not feature.contours:
            return None
        
        # Get main contour
        main_contour = max(feature.contours, key=cv2.contourArea)
        contour_points = main_contour.reshape(-1, 2).astype(np.float32)
        
        # Find base and tip
        anchor_array = np.array(base_anchor, dtype=np.float32)
        distances_to_anchor = np.linalg.norm(contour_points - anchor_array, axis=1)
        base_idx = np.argmin(distances_to_anchor)
        base_point = contour_points[base_idx]
        
        distances_from_base = np.linalg.norm(contour_points - base_point, axis=1)
        tip_idx = np.argmax(distances_from_base)
        
        # Split contour into two sides
        n = len(contour_points)
        
        if base_idx < tip_idx:
            side1 = contour_points[base_idx:tip_idx+1]
            side2 = np.vstack([contour_points[tip_idx:], contour_points[:base_idx+1]])
        else:
            side1 = np.vstack([contour_points[base_idx:], contour_points[:tip_idx+1]])
            side2 = contour_points[tip_idx:base_idx+1]
        
        # Compute gradient strength for each side
        print("  Analyzing edge gradients...")
        gradient_strength1 = self._compute_edge_gradient_strength(side1)
        gradient_strength2 = self._compute_edge_gradient_strength(side2)
        
        print(f"  Side 1 gradient strength: {gradient_strength1:.2f}")
        print(f"  Side 2 gradient strength: {gradient_strength2:.2f}")
        
        # Select side with stronger gradient (front edge)
        if gradient_strength1 > gradient_strength2:
            selected_side = side1
            ratio = gradient_strength1 / gradient_strength2 if gradient_strength2 > 0 else float('inf')
            print(f"  → Selected Side 1 (stronger gradient = front edge, ratio: {ratio:.2f}x)")
        else:
            selected_side = side2[::-1]  # Reverse to go base->tip
            ratio = gradient_strength2 / gradient_strength1 if gradient_strength1 > 0 else float('inf')
            print(f"  → Selected Side 2 (stronger gradient = front edge, ratio: {ratio:.2f}x)")
        
        print(f"  Contour extracted: {len(selected_side)} points from base to tip")
        
        return selected_side
    
    def _compute_edge_gradient_strength(self, edge_points: np.ndarray) -> float:
        """
        Compute average gradient magnitude perpendicular to edge.
        
        For each point on the edge:
        1. Compute tangent direction (along edge)
        2. Compute perpendicular (normal) direction (pointing outward)
        3. Sample gradient in normal direction
        4. Return average magnitude
        
        Args:
            edge_points: Array of (N, 2) edge coordinates
            
        Returns:
            Average gradient magnitude perpendicular to edge
        """
        if len(edge_points) < 3:
            return 0.0
        
        # Compute image gradients using Sobel
        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitudes = []
        
        # Sample every few points for efficiency (not every single point)
        step = max(1, len(edge_points) // 50)  # Sample ~50 points
        
        for i in range(1, len(edge_points) - 1, step):
            # Get local tangent (from previous to next point)
            tangent = edge_points[i+1] - edge_points[i-1]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm < 1e-6:
                continue
            
            tangent = tangent / tangent_norm
            
            # Perpendicular (normal) direction - rotate tangent 90° counterclockwise
            normal = np.array([-tangent[1], tangent[0]])
            
            # Get gradient at this point
            x, y = int(edge_points[i, 0]), int(edge_points[i, 1])
            
            # Bounds check
            if x < 0 or x >= self.W or y < 0 or y >= self.H:
                continue
            
            gx = grad_x[y, x]
            gy = grad_y[y, x]
            gradient_vec = np.array([gx, gy])
            
            # Project gradient onto normal direction
            # Use absolute value because we care about magnitude, not direction
            gradient_normal = np.abs(np.dot(gradient_vec, normal))
            
            gradient_magnitudes.append(gradient_normal)
        
        if len(gradient_magnitudes) == 0:
            return 0.0
        
        # Return average gradient magnitude
        return float(np.mean(gradient_magnitudes))
        
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
    
    def fit_box_to_feature(self, marker_id: int, method: str = 'ransac') -> Optional[np.ndarray]:
        """
        Fit a rectangular box to a feature using various algorithms.
        
        Args:
            marker_id: Feature marker ID
            method: Box fitting method - 'ransac', 'pca', or 'minarea'
                - 'ransac': Hough Transform + RANSAC (best for burrs, rounded corners)
                - 'pca': PCA on edge pixels (good for clean rectangles)
                - 'minarea': Minimum area rectangle (fastest, sensitive to corners)
        
        Returns:
            Array of 4 corners [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            Sorted as: top-left, top-right, bottom-right, bottom-left
        """
        if marker_id not in self.features:
            return None
        
        if method == 'ransac':
            return self._fit_box_ransac(marker_id)
        elif method == 'pca':
            return self._fit_box_pca(marker_id)
        elif method == 'minarea':
            return self._fit_box_minarea(marker_id)
        else:
            raise ValueError(f"Unknown box fitting method: {method}. Use 'ransac', 'pca', or 'minarea'")
    
    def _fit_box_ransac(self, marker_id: int) -> Optional[np.ndarray]:
        """
        Fit box using Hough Transform + RANSAC for edge detection.
        
        Robust to rounded corners, burrs, and noise.
        """
        feature = self.features[marker_id]
        mask = feature.mask
        
        # Edge detection
        edges = cv2.Canny(mask, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                                threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            print("  RANSAC: Not enough edges, falling back to minarea")
            return self._fit_box_minarea(marker_id)
        
        # Extract angles from lines
        angles = []
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
            line_data.append(line[0])
        
        angles = np.array(angles)
        line_data = np.array(line_data)
        
        # Normalize angles to [0, π)
        angles_norm = angles % np.pi
        
        # Find two dominant perpendicular orientations
        hist, bins = np.histogram(angles_norm, bins=36)  # 5-degree bins
        
        # Find peaks in histogram
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, distance=15, height=2)
        
        if len(peaks) < 2:
            print("  RANSAC: Could not find perpendicular edges, falling back")
            return self._fit_box_minarea(marker_id)
        
        # Take two strongest peaks
        peak_heights = hist[peaks]
        top_2_idx = np.argsort(peak_heights)[-2:]
        top_2_peaks = peaks[top_2_idx]
        
        dominant_angles = bins[top_2_peaks]
        
        # For each dominant angle, find extreme parallel lines
        box_lines = []
        
        for dominant_angle in dominant_angles:
            # Find lines close to this angle
            angle_diff = np.abs((angles_norm - dominant_angle + np.pi/2) % np.pi - np.pi/2)
            close_mask = angle_diff < np.pi/18  # Within 10 degrees
            
            if np.sum(close_mask) == 0:
                continue
            
            close_lines = line_data[close_mask]
            
            # Extract all points from these lines
            points = []
            for x1, y1, x2, y2 in close_lines:
                points.extend([[x1, y1], [x2, y2]])
            points = np.array(points)
            
            # Project points onto perpendicular direction
            perp_angle = dominant_angle + np.pi/2
            perp_vec = np.array([np.cos(perp_angle), np.sin(perp_angle)])
            
            projections = points @ perp_vec
            
            # Find min and max projections (the two parallel edges)
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            
            edge_vec = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
            
            box_lines.append({
                'angle': dominant_angle,
                'projections': [min_proj, max_proj],
                'direction': edge_vec,
                'perpendicular': perp_vec
            })
        
        if len(box_lines) < 2:
            print("  RANSAC: Could not find two edge pairs")
            return self._fit_box_minarea(marker_id)
        
        # Compute box corners from line intersections
        edges1 = box_lines[0]
        edges2 = box_lines[1]
        
        corners = []
        for proj1 in edges1['projections']:
            for proj2 in edges2['projections']:
                # Intersection of two lines
                p1 = proj1 * edges1['perpendicular']
                p2 = proj2 * edges2['perpendicular']
                
                # Solve: p1 + t1 * dir1 = p2 + t2 * dir2
                A = np.column_stack([edges1['direction'], -edges2['direction']])
                b = p2 - p1
                
                try:
                    t = np.linalg.solve(A, b)
                    corner = p1 + t[0] * edges1['direction']
                    corners.append(corner)
                except np.linalg.LinAlgError:
                    continue
        
        if len(corners) != 4:
            print(f"  RANSAC: Expected 4 corners, got {len(corners)}, falling back")
            return self._fit_box_minarea(marker_id)
        
        corners = np.array(corners)
        corners = self._sort_box_corners(corners)
        
        print("  ✓ RANSAC box fitting successful")
        return corners
    
    def _fit_box_pca(self, marker_id: int) -> Optional[np.ndarray]:
        """
        Fit box using PCA on edge pixels.
        
        Good for clean rectangles with minor corner issues.
        """
        feature = self.features[marker_id]
        
        # Get edge pixels
        edges = cv2.Canny(feature.mask, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))[:, ::-1]  # (x, y)
        
        if len(edge_points) < 100:
            print("  PCA: Not enough edge points, falling back")
            return self._fit_box_minarea(marker_id)
        
        # PCA to find principal axes
        mean = edge_points.mean(axis=0)
        centered = edge_points - mean
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project points onto principal axes
        projected = centered @ eigenvectors
        
        # Find bounding box in PCA space
        min_proj = projected.min(axis=0)
        max_proj = projected.max(axis=0)
        
        # Corners in PCA space
        corners_pca = np.array([
            [min_proj[0], min_proj[1]],
            [max_proj[0], min_proj[1]],
            [max_proj[0], max_proj[1]],
            [min_proj[0], max_proj[1]]
        ])
        
        # Transform back to image space
        corners = corners_pca @ eigenvectors.T + mean
        corners = self._sort_box_corners(corners)
        
        print("  ✓ PCA box fitting successful")
        return corners
    
    def _fit_box_minarea(self, marker_id: int) -> Optional[np.ndarray]:
        """
        Fit box using cv2.minAreaRect (minimum area bounding rectangle).
        
        Fastest but sensitive to corner artifacts.
        """
        feature = self.features[marker_id]
        
        if not feature.contours:
            return None
        
        # Get main contour
        main_contour = max(feature.contours, key=cv2.contourArea)
        
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        
        corners = self._sort_box_corners(box)
        
        print("  ✓ MinArea box fitting successful")
        return corners
    
    def _sort_box_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Sort box corners to standard order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corners (any order)
            
        Returns:
            Array of 4 corners in standard order
        """
        # Sort by y-coordinate
        corners_sorted = corners[np.argsort(corners[:, 1])]
        
        # Top two points (lowest y values)
        top_points = corners_sorted[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Sort by x
        
        # Bottom two points (highest y values)
        bottom_points = corners_sorted[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x
        
        # Arrange as [top-left, top-right, bottom-right, bottom-left]
        return np.array([
            top_points[0],      # Top-left
            top_points[1],      # Top-right
            bottom_points[1],   # Bottom-right
            bottom_points[0]    # Bottom-left
        ])
    
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
