"""
Integrated workflow: Watershed segmentation → Box fitting → Curve model fitting → Tracking
"""

from typing import Optional, Tuple, List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba
import cv2

from segmentvideo.utils.watershed import WatershedSegmenter
from segmentvideo.models.curve import CurveModel
from segmentvideo.annotation.state import AnnotationState


class IntegratedSegmentationWorkflow:
    """
    Manages the complete workflow from watershed segmentation to curve tracking.
    
    Workflow:
    1. User places seeds on frame 0
    2. Run watershed segmentation → automatically fits box to Feature 2
    3. User adjusts box corners if needed
    4. User selects edge point on feature to indicate tracking edge
    5. Fit curve model to selected edge with anchor at box top edge midpoint
    """
    
    def __init__(self, image: np.ndarray, n_curve_points: int = 10):
        """
        Initialize workflow.
        
        Args:
            image: First frame as (H, W, 3) RGB array
            n_curve_points: Number of control points for curve model
        """
        self.image = image
        self.H, self.W = image.shape[:2]
        self.n_curve_points = n_curve_points
        
        # Initialize watershed segmenter
        self.segmenter = WatershedSegmenter(image)
        
        # Workflow state
        self.step = 0  # 0=seed_placement, 1=box_fitting, 2=edge_selection, 3=curve_fitting, 4=complete
        self.step_names = [
            'Seed Placement',
            'Box Adjustment', 
            'Edge Selection',
            'Curve Fitting',
            'Complete'
        ]
        self.current_feature_id = 2  # Start at 2 (1 is background)
        self.selected_edge_point: Optional[Tuple[float, float]] = None
        self.curve_model: Optional[CurveModel] = None
        
        # Box fitting
        self.box_corners: Optional[np.ndarray] = None  # 4 corners of the fitted box
        self.base_anchor: Optional[Tuple[float, float]] = None  # Midpoint of top edge
        self.active_corner: Optional[int] = None  # Currently dragged corner
        
        # UI components
        self.fig = None
        self.ax_main = None
        self.ax_debug_1 = None
        self.ax_debug_2 = None
        self.seed_patches: List[Circle] = []
        self.active_seed = None
        self.press_pos = None
    
    def run_interactive(self):
        """Launch the interactive UI for the complete workflow."""
        self._setup_ui()
        plt.show()
    
    def _setup_ui(self):
        """Setup matplotlib UI."""
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(2, 4)
        
        # Main axis for seed placement and segmentation
        self.ax_main = self.fig.add_subplot(gs[:, :3])
        self.ax_debug_1 = self.fig.add_subplot(gs[0, 3])
        self.ax_debug_2 = self.fig.add_subplot(gs[1, 3])
        
        self.ax_main.imshow(self.image)
        self._update_display()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
        # Setup radio buttons for seed selection
        ax_radio = plt.axes([0.02, 0.4, 0.08, 0.2])
        self.radio = RadioButtons(ax_radio, ('Background', 'Feature 1', 'Feature 2'))
        self.radio.on_clicked(self._set_active_id)
        # Set Feature 1 as active by default (index 1)
        self.radio.set_active(1)  # 0=Background, 1=Feature 1, 2=Feature 2
        
        # Simplified navigation buttons
        self.btn_prev = Button(plt.axes([0.15, 0.02, 0.15, 0.05]), '← Previous Step', color='lightgray')
        self.btn_prev.on_clicked(self._prev_step)
        
        self.btn_next = Button(plt.axes([0.35, 0.02, 0.15, 0.05]), 'Next Step →', color='lightgreen')
        self.btn_next.on_clicked(self._next_step)
    
    def _update_display(self):
        """Update the display based on current step."""
        # Clear main axis
        self.ax_main.clear()
        self.ax_main.imshow(self.image)
        
        # Update title with step indicator
        step_text = f"STEP {self.step + 1}/5: {self.step_names[self.step]}"
        
        if self.step == 0:  # Seed placement
            instructions = "Click to add seeds | Drag to move | Click seed to remove"
            self.ax_main.set_title(f"{step_text}\n{instructions}")
            
            # Re-draw seeds
            for seed, patch in zip(self.segmenter.seeds, self.seed_patches):
                self.ax_main.add_patch(patch)
        
        elif self.step == 1:  # Box fitting
            instructions = "Drag blue corners to adjust box | Green star = anchor point"
            self.ax_main.set_title(f"{step_text}\n{instructions}")
            
            # Show watershed overlay
            overlay = self.segmenter.get_overlay(alpha=0.3)
            self.ax_main.imshow(overlay)
            
            # Draw box if it exists
            if self.box_corners is not None:
                # Draw box (closed polygon)
                box_closed = np.vstack([self.box_corners, self.box_corners[0]])
                self.ax_main.plot(box_closed[:, 0], box_closed[:, 1], 'b-', linewidth=3, label='Box')
                
                # Draw corners as draggable points
                self.ax_main.plot(self.box_corners[:, 0], self.box_corners[:, 1], 
                                 'bs', markersize=12, markeredgecolor='white', markeredgewidth=2)
                
                # Draw anchor point (midpoint of top edge)
                if self.base_anchor:
                    self.ax_main.plot(self.base_anchor[0], self.base_anchor[1], 
                                     'g*', markersize=20, markeredgecolor='white', markeredgewidth=2)
                
                # Add text labels
                self.ax_main.text(self.base_anchor[0], self.base_anchor[1] - 20, 
                                 'ANCHOR', color='green', fontsize=12, fontweight='bold',
                                 ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        elif self.step == 2:  # Edge selection
            instructions = "Click on the LEFT or RIGHT edge of Feature 1 (needle) to track"
            self.ax_main.set_title(f"{step_text}\n{instructions}")
            
            # Show watershed overlay
            overlay = self.segmenter.get_overlay(alpha=0.3)
            self.ax_main.imshow(overlay)
            
            # Show box (faded)
            if self.box_corners is not None:
                box_closed = np.vstack([self.box_corners, self.box_corners[0]])
                self.ax_main.plot(box_closed[:, 0], box_closed[:, 1], 'b--', 
                                 linewidth=1, alpha=0.5)
                if self.base_anchor:
                    self.ax_main.plot(self.base_anchor[0], self.base_anchor[1], 
                                     'g*', markersize=15, alpha=0.5)
            
            # Show selected edge point if exists
            if self.selected_edge_point:
                self.ax_main.plot(self.selected_edge_point[0], self.selected_edge_point[1], 
                                 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        
        elif self.step == 3:  # Curve fitting
            instructions = "Review the fitted curve | Click 'Next Step' when satisfied"
            self.ax_main.set_title(f"{step_text}\n{instructions}")
            
            # Show watershed overlay (faded)
            overlay = self.segmenter.get_overlay(alpha=0.1)
            self.ax_main.imshow(overlay)
            
            # Show curve model if fitted
            if self.curve_model:
                self.curve_model.render(self.ax_main, color='red', linewidth=3, 
                                       show_backbone=True)
                
                # Show anchor
                if self.base_anchor:
                    self.ax_main.plot(self.base_anchor[0], self.base_anchor[1], 
                                     'g*', markersize=15, alpha=0.7)
        
        elif self.step == 4:  # Complete
            instructions = "Workflow complete! Close window to finish."
            self.ax_main.set_title(f"{step_text}\n{instructions}")
            
            # Show final result
            if self.curve_model:
                self.curve_model.render(self.ax_main, color='red', linewidth=3)
        
        self.fig.canvas.draw_idle()
    
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax_main:
            return
        
        if self.step == 0:  # Seed placement
            self._handle_seed_press(event)
        elif self.step == 1:  # Box fitting
            self._handle_box_press(event)
        elif self.step == 2:  # Edge selection
            self._handle_edge_selection(event)
    
    def _on_release(self, event):
        """Handle mouse release events."""
        if self.step == 0:
            self._handle_seed_release(event)
        elif self.step == 1:
            self._handle_box_release(event)
    
    def _on_motion(self, event):
        """Handle mouse motion events."""
        if self.step == 0:
            self._handle_seed_motion(event)
        elif self.step == 1:
            self._handle_box_motion(event)
    
    def _handle_seed_press(self, event):
        """Handle mouse press in seed placement mode."""
        # Check if clicking an existing seed
        for i, patch in enumerate(self.seed_patches):
            if patch.contains(event)[0]:
                self.active_seed = (patch, i)
                self.press_pos = (event.xdata, event.ydata)
                return
        
        # Add new seed if no seed was clicked
        if event.button == 1:  # Left click
            seed = self.segmenter.add_seed(event.xdata, event.ydata, self.current_feature_id)
            patch = Circle((event.xdata, event.ydata), 
                          radius=self.W // 80, 
                          color=seed.color, 
                          alpha=0.7)
            self.ax_main.add_patch(patch)
            self.seed_patches.append(patch)
            self.fig.canvas.draw_idle()
    
    def _handle_seed_motion(self, event):
        """Handle mouse motion in seed placement mode (for dragging)."""
        if self.active_seed is None or event.inaxes != self.ax_main:
            return
        
        patch, idx = self.active_seed
        patch.center = (event.xdata, event.ydata)
        
        # Update seed position
        self.segmenter.seeds[idx].x = event.xdata
        self.segmenter.seeds[idx].y = event.ydata
        
        self.fig.canvas.draw_idle()
    
    def _handle_seed_release(self, event):
        """Handle mouse release in seed placement mode."""
        if self.active_seed and self.press_pos:
            # If the mouse didn't move much, treat it as a deletion click
            dist = np.sqrt((event.xdata - self.press_pos[0]) ** 2 + 
                          (event.ydata - self.press_pos[1]) ** 2)
            
            if dist < self.W // 200:  # Small movement = delete
                patch, idx = self.active_seed
                patch.remove()
                self.seed_patches.pop(idx)
                self.segmenter.remove_seed(idx)
        
        self.active_seed = None
        self.press_pos = None
        self.fig.canvas.draw_idle()
    
    def _handle_box_press(self, event):
        """Handle mouse press in box fitting mode."""
        if self.box_corners is None:
            return
        
        # Check if clicking near a corner
        click_point = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.box_corners - click_point, axis=1)
        
        if np.min(distances) < 30:  # Within 30 pixels
            self.active_corner = np.argmin(distances)
            print(f"Grabbed corner {self.active_corner}")
        else:
            self.active_corner = None
    
    def _handle_box_motion(self, event):
        """Handle mouse motion in box fitting mode (dragging corners)."""
        if self.active_corner is None or event.inaxes != self.ax_main:
            return
        
        # Update corner position
        self.box_corners[self.active_corner] = [event.xdata, event.ydata]
        
        # Recalculate anchor (midpoint of top edge - between corners 0 and 1)
        self.base_anchor = tuple(((self.box_corners[0] + self.box_corners[1]) / 2).astype(float))
        
        # Redraw
        self._update_display()
    
    def _handle_box_release(self, event):
        """Handle mouse release in box fitting mode."""
        if self.active_corner is not None:
            print(f"Released corner {self.active_corner}")
            print(f"New anchor position: ({self.base_anchor[0]:.1f}, {self.base_anchor[1]:.1f})")
        self.active_corner = None
    
    def _handle_edge_selection(self, event):
        """Handle mouse click in edge selection mode."""
        if event.button == 1:  # Left click
            self.selected_edge_point = (event.xdata, event.ydata)
            print(f"Edge point selected at ({event.xdata:.1f}, {event.ydata:.1f})")
            self._update_display()
    
    def _set_active_id(self, label: str):
        """Set the current feature ID based on radio button selection."""
        if label == "Background":
            self.current_feature_id = 1
        else:
            feature_num = int(label.split(" ")[1])
            self.current_feature_id = feature_num + 1  # Feature 1 -> 2, Feature 2 -> 3
        print(f"Active feature ID: {self.current_feature_id}")
    
    def _prev_step(self, event):
        """Go to previous step."""
        if self.step > 0:
            self.step -= 1
            print(f"\n← Previous: {self.step_names[self.step]}")
            self._update_display()
    
    def _next_step(self, event):
        """Go to next step."""
        if self.step == 0:  # Seed placement -> Watershed + Box fitting
            if not self.segmenter.seeds:
                print("❌ No seeds placed. Add seeds before continuing.")
                return
            
            print("\n→ Running watershed segmentation...")
            try:
                markers = self.segmenter.run_watershed(blur_kernel=5)
                print(f"✓ Watershed completed. Found {len(self.segmenter.features)} features.")
                
                # Fit box to Feature 2
                self._fit_box_to_feature2()
                
                self.step = 1
                print(f"→ Next: {self.step_names[self.step]}")
                self._update_display()
                
            except Exception as e:
                print(f"❌ Error running watershed: {e}")
        
        elif self.step == 1:  # Box fitting -> Edge selection
            if self.box_corners is None or self.base_anchor is None:
                print("❌ Box not fitted properly.")
                return
            
            print(f"✓ Box confirmed. Anchor at ({self.base_anchor[0]:.1f}, {self.base_anchor[1]:.1f})")
            self.step = 2
            print(f"→ Next: {self.step_names[self.step]}")
            self._update_display()
        
        elif self.step == 2:  # Edge selection -> Curve fitting
            if self.selected_edge_point is None:
                print("❌ No edge point selected. Click on Feature 1 edge.")
                return
            
            if self.base_anchor is None:
                print("❌ Box anchor not set.")
                return
            
            print("\n→ Fitting curve to edge...")
            self._fit_curve_to_edge()
        
        elif self.step == 3:  # Curve fitting -> Complete
            print("\n✓ Workflow complete!")
            self.step = 4
            self._update_display()
        
        elif self.step == 4:  # Complete
            print("Workflow already complete. Close window to finish.")
    
    def _fit_box_to_feature2(self):
        """Fit a rectangular box to Feature 2 (the base block)."""
        if 3 not in self.segmenter.features:  # Feature 2 is marker_id=3
            print("⚠ Feature 2 not found. Cannot fit box.")
            return
        
        feature = self.segmenter.features[3]
        
        if not feature.contours:
            print("⚠ No contours found for Feature 2.")
            return
        
        # Get main contour
        main_contour = max(feature.contours, key=cv2.contourArea)
        
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        # Sort by y-coordinate
        box_sorted = box[np.argsort(box[:, 1])]
        
        # Top two points (lowest y values - at top of image)
        top_points = box_sorted[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Sort by x
        
        # Bottom two points (highest y values - at bottom of image)
        bottom_points = box_sorted[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x
        
        # Arrange as [top-left, top-right, bottom-right, bottom-left]
        self.box_corners = np.array([
            top_points[0],      # Top-left
            top_points[1],      # Top-right
            bottom_points[1],   # Bottom-right
            bottom_points[0]    # Bottom-left
        ])
        
        # Calculate anchor point (midpoint of top edge)
        self.base_anchor = tuple(((self.box_corners[0] + self.box_corners[1]) / 2).astype(float))
        
        print(f"✓ Box fitted to Feature 2")
        print(f"  Anchor at: ({self.base_anchor[0]:.1f}, {self.base_anchor[1]:.1f})")
    
    def _fit_curve_to_edge(self):
        """Fit a curve model to the selected edge."""
        # Extract edge contour for Feature 1 (marker_id = 2) with anchor
        contour = self.segmenter.get_feature_edge_contour_with_anchor(
            marker_id=2,
            edge_point=self.selected_edge_point,
            base_anchor=self.base_anchor
        )
        
        if contour is None:
            print("❌ Could not extract contour for Feature 1.")
            return
        
        # Calculate anchor tangent (perpendicular to box top edge)
        # Box top edge is from corner 0 to corner 1
        if self.box_corners is not None:
            box_edge = self.box_corners[1] - self.box_corners[0]
            box_edge_normalized = box_edge / np.linalg.norm(box_edge)
            # Perpendicular: rotate 90 degrees counterclockwise
            anchor_tangent = np.array([-box_edge_normalized[1], box_edge_normalized[0]])
            print(f"  Anchor tangent: ({anchor_tangent[0]:.3f}, {anchor_tangent[1]:.3f})")
        else:
            anchor_tangent = None
        
        # Create curve model from contour using B-spline
        self.curve_model = CurveModel.from_contour(
            frame_idx=0,
            contour_points=contour,
            n_points=self.n_curve_points,
            use_bspline=True,  # Use B-spline for optimal representation
            smoothing=0.0,     # No smoothing (interpolates through all points)
            anchor_tangent=anchor_tangent
        )
        
        print(f"✓ Curve model fitted with {self.n_curve_points} control points")
        print(f"  Spline type: {self.curve_model.spline_type}")
        print(f"  Total length: {self.curve_model.get_total_length():.1f} px")
        print(f"  Backbone length: {self.curve_model.get_backbone_length():.1f} px")
        
        # Compute curvature at a few sample points
        t_sample = np.linspace(0, 1, 5)
        curvature_sample = self.curve_model.compute_curvature(t_sample)
        print(f"  Curvature (sample): {curvature_sample}")
        
        # Move to next step
        self.step = 3
        print(f"→ Next: {self.step_names[self.step]}")
        self._update_display()
    
    def get_curve_model(self) -> Optional[CurveModel]:
        """Get the fitted curve model."""
        return self.curve_model
    
    def export_curve_parameters(self) -> Optional[dict]:
        """Export curve model parameters for tracking."""
        if self.curve_model is None:
            return None
        
        return self.curve_model.to_dict()


def run_segmentation_workflow(image: np.ndarray, n_curve_points: int = 10) -> Optional[CurveModel]:
    """
    Run the interactive segmentation workflow.
    
    Args:
        image: First frame as (H, W, 3) RGB array
        n_curve_points: Number of control points for curve model
        
    Returns:
        Fitted CurveModel or None if workflow was cancelled
    """
    workflow = IntegratedSegmentationWorkflow(image, n_curve_points)
    workflow.run_interactive()
    return workflow.get_curve_model()
