"""
Integrated workflow: Watershed segmentation → Curve model fitting → Tracking
"""

from typing import Optional, Tuple, List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba

from segmentvideo.utils.watershed import WatershedSegmenter
from segmentvideo.models.curve import CurveModel
from segmentvideo.annotation.state import AnnotationState


class IntegratedSegmentationWorkflow:
    """
    Manages the complete workflow from watershed segmentation to curve tracking.
    
    Workflow:
    1. User places seeds on frame 0
    2. Run watershed segmentation
    3. User selects edge point on feature to indicate tracking edge
    4. Fit curve model to selected edge
    5. Track curve through subsequent frames with user verification
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
        self.state = 'seed_placement'  # seed_placement, edge_selection, tracking
        self.current_feature_id = 2  # Start at 2 (1 is background)
        self.selected_edge_point: Optional[Tuple[float, float]] = None
        self.curve_model: Optional[CurveModel] = None
        
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
        self._update_title()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
        # Setup buttons - store references to prevent garbage collection
        ax_radio = plt.axes([0.02, 0.4, 0.08, 0.2])
        self.radio = RadioButtons(ax_radio, ('Background', 'Feature 1', 'Feature 2'))
        self.radio.on_clicked(self._set_active_id)
        # Set Feature 1 as active by default (index 1)
        self.radio.set_active(1)  # 0=Background, 1=Feature 1, 2=Feature 2
        
        self.btn_watershed = Button(plt.axes([0.15, 0.02, 0.12, 0.05]), 'Update Mask', color='lime')
        self.btn_watershed.on_clicked(self._run_watershed)
        
        self.btn_fit_curve = Button(plt.axes([0.3, 0.02, 0.12, 0.05]), 'Fit Curve', color='orange')
        self.btn_fit_curve.on_clicked(self._fit_curve_to_edge)
        
        self.btn_next = Button(plt.axes([0.45, 0.02, 0.12, 0.05]), 'Next Step', color='cyan')
        self.btn_next.on_clicked(self._next_step)
    
    def _update_title(self):
        """Update main axis title based on workflow state."""
        if self.state == 'seed_placement':
            title = "STEP 1: Click to add seeds | Drag to move | Click point to remove"
        elif self.state == 'edge_selection':
            title = "STEP 2: Click on the edge of Feature 1 that you want to track"
        elif self.state == 'tracking':
            title = "STEP 3: Curve model fitted! Ready for tracking through frames"
        else:
            title = "Segmentation Workflow"
        
        self.ax_main.set_title(title)
    
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax_main:
            return
        
        if self.state == 'seed_placement':
            self._handle_seed_press(event)
        elif self.state == 'edge_selection':
            self._handle_edge_selection(event)
    
    def _on_release(self, event):
        """Handle mouse release events."""
        if self.state == 'seed_placement':
            self._handle_seed_release(event)
    
    def _on_motion(self, event):
        """Handle mouse motion events."""
        if self.state == 'seed_placement':
            self._handle_seed_motion(event)
    
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
            self.fig.canvas.draw()
    
    def _handle_seed_motion(self, event):
        """Handle mouse motion in seed placement mode (for dragging)."""
        if self.active_seed is None or event.inaxes != self.ax_main:
            return
        
        patch, idx = self.active_seed
        patch.center = (event.xdata, event.ydata)
        
        # Update seed position
        self.segmenter.seeds[idx].x = event.xdata
        self.segmenter.seeds[idx].y = event.ydata
        
        self.fig.canvas.draw()
    
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
        self.fig.canvas.draw()
    
    def _handle_edge_selection(self, event):
        """Handle mouse click in edge selection mode."""
        if event.button == 1:  # Left click
            self.selected_edge_point = (event.xdata, event.ydata)
            
            # Draw a marker at the selected point
            self.ax_main.plot(event.xdata, event.ydata, 'r*', markersize=15)
            self.fig.canvas.draw()
            
            print(f"Edge point selected at ({event.xdata:.1f}, {event.ydata:.1f})")
    
    def _set_active_id(self, label: str):
        """Set the current feature ID based on radio button selection."""
        if label == "Background":
            self.current_feature_id = 1
        else:
            feature_num = int(label.split(" ")[1])
            self.current_feature_id = feature_num + 1  # Feature 1 -> 2, Feature 2 -> 3
    
    def _run_watershed(self, event):
        """Run watershed segmentation."""
        if not self.segmenter.seeds:
            print("No seeds placed. Add seeds before running watershed.")
            return
        
        try:
            markers = self.segmenter.run_watershed(blur_kernel=5)
            
            # Display result
            overlay = self.segmenter.get_overlay(alpha=0.4)
            
            self.ax_main.clear()
            self.ax_main.imshow(self.image)
            self.ax_main.imshow(overlay)
            
            # Re-draw seeds
            for seed, patch in zip(self.segmenter.seeds, self.seed_patches):
                self.ax_main.add_patch(patch)
            
            self._update_title()
            self.fig.canvas.draw()
            
            print(f"Watershed completed. Found {len(self.segmenter.features)} features.")
            
            # Move to edge selection state
            self.state = 'edge_selection'
            self._update_title()
            
        except Exception as e:
            print(f"Error running watershed: {e}")
    
    def _fit_curve_to_edge(self, event):
        """Fit a curve model to the selected edge."""
        if self.selected_edge_point is None:
            print("No edge point selected. Click on Feature 1 edge first.")
            return
        
        # Extract edge contour for Feature 1 (marker_id = 2)
        contour = self.segmenter.get_feature_edge_contour(2, self.selected_edge_point)
        
        if contour is None:
            print("Could not extract contour for Feature 1.")
            return
        
        # Create curve model from contour
        self.curve_model = CurveModel.from_contour(
            frame_idx=0,
            contour_points=contour,
            n_points=self.n_curve_points
        )
        
        # Display curve model
        self.ax_main.clear()
        self.ax_main.imshow(self.image)
        
        # Show watershed overlay
        overlay = self.segmenter.get_overlay(alpha=0.2)
        self.ax_main.imshow(overlay)
        
        # Render curve model
        self.curve_model.render(self.ax_main, color='red', linewidth=3)
        
        self._update_title()
        self.fig.canvas.draw()
        
        print(f"Curve model fitted with {self.n_curve_points} points")
        print(f"Total curve length: {self.curve_model.get_total_length():.1f} pixels")
        
        # Move to tracking state
        self.state = 'tracking'
        self._update_title()
    
    def _next_step(self, event):
        """Move to the next step in the workflow."""
        if self.state == 'seed_placement':
            print("Run watershed first (click 'Update Mask')")
        elif self.state == 'edge_selection':
            print("Select an edge point, then click 'Fit Curve'")
        elif self.state == 'tracking':
            print("Workflow complete! Curve model is ready for tracking.")
            print("You can now use this curve model as the initial prediction for frame-by-frame tracking.")
    
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
