import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

# 1. Load and Pre-process
img_path = "test_img.png"
try:
    img_raw = plt.imread(f"../videos/{img_path}")
except:
    img_raw = np.zeros((500, 500, 3), dtype=np.float32)
    cv2.line(img_raw, (50, 50), (200, 200), (0.8, 0.2, 0.2), 15)  # Strip 1
    cv2.line(img_raw, (300, 100), (450, 400), (0.2, 0.8, 0.2), 15)  # Strip 2
    img_raw = np.clip(img_raw + np.random.normal(0, 0.02, img_raw.shape), 0, 1)

if img_raw.max() <= 1.0:
    img_8bit = (img_raw[:, :, :3] * 255).astype(np.uint8)
else:
    img_8bit = img_raw[:, :, :3].astype(np.uint8)

img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
H, W, _ = img_bgr.shape


# 2. Interactive Seed Manager
class MultiFeatureManager:
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.seeds = []
        self.current_feature_id = 2  # Start at 2 (1 is background)
        self.active_seed = None
        self.press = None

        # Colors for different features
        self.color_cycle = ['lime', 'magenta', 'yellow', 'orange', 'cyan', 'white']

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        # Drag existing seed
        for patch, m_id in self.seeds:
            contains, _ = patch.contains(event)
            if contains:
                self.active_seed = patch
                self.press = patch.center, event.xdata, event.ydata
                return

        # Add new seed
        if event.button == 1:  # Left click: Current Feature
            m_id = self.current_feature_id
            color = self.color_cycle[(m_id - 2) % len(self.color_cycle)]
        elif event.button == 3:  # Right click: Background
            m_id = 1
            color = 'deepskyblue'
        else:
            return

        new_patch = plt.Circle((event.xdata, event.ydata), radius=W // 60, color=color, alpha=0.7)
        self.ax.add_patch(new_patch)
        self.seeds.append((new_patch, m_id))
        self.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax: return
        (x0, y0), xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.active_seed.center = (x0 + dx, y0 + dy)
        self.canvas.draw()

    def on_release(self, event):
        self.active_seed = None
        self.press = None
        self.canvas.draw()

    def next_feature(self):
        self.current_feature_id += 1
        print(f"Now adding seeds for Feature ID: {self.current_feature_id}")

    def clear_seeds(self):
        for patch, _ in self.seeds: patch.remove()
        self.seeds = []
        self.current_feature_id = 2
        self.canvas.draw()

    def get_seed_data(self):
        return [(int(p.center[0]), int(p.center[1]), m_id) for p, m_id in self.seeds]


# 3. UI and Plotting
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8))
plt.subplots_adjust(bottom=0.25)

ax_left.imshow(img_raw)
ax_left.set_title("L-Click: Add Seed | R-Click: Background\nUse 'New Feature' to switch IDs")
ax_right.imshow(img_raw)
ax_right.set_title("Watershed Result")

manager = MultiFeatureManager(ax_left)


# 4. Actions
def calculate_watershed(event):
    seed_data = manager.get_seed_data()
    markers = np.zeros((H, W), dtype=np.int32)
    # Background border
    markers[0, :], markers[-1, :], markers[:, 0], markers[:, -1] = 1, 1, 1, 1

    for sx, sy, m_id in seed_data:
        if 0 <= sy < H and 0 <= sx < W:
            cv2.circle(markers, (sx, sy), 3, m_id, -1)

    markers_res = cv2.watershed(cv2.GaussianBlur(img_bgr, (5, 5), 0), markers)

    # Render multi-colored mask
    # We use a colormap to visualize different IDs differently
    res_viz = np.zeros((H, W, 4))
    unique_ids = np.unique(markers_res)
    for m_id in unique_ids:
        if m_id > 1:  # Skip background and boundaries
            color = plt.cm.get_cmap('tab10')(m_id % 10)
            res_viz[markers_res == m_id] = [*color[:3], 0.5]

    ax_right.clear()
    ax_right.imshow(img_raw)
    ax_right.imshow(res_viz)
    ax_right.set_title("Multi-Feature Segmentation")
    fig.canvas.draw_idle()


# Buttons
ax_next = plt.axes([0.15, 0.05, 0.15, 0.075])
btn_next = Button(ax_next, 'Add New Feature', color='lightgray', hovercolor='plum')
btn_next.on_clicked(lambda e: manager.next_feature())

ax_calc = plt.axes([0.4, 0.05, 0.15, 0.075])
btn_calc = Button(ax_calc, 'Calculate', color='lightgray', hovercolor='lime')
btn_calc.on_clicked(calculate_watershed)

ax_clear = plt.axes([0.65, 0.05, 0.15, 0.075])
btn_clear = Button(ax_clear, 'Clear All', color='lightgray', hovercolor='tomato')
btn_clear.on_clicked(
    lambda e: [manager.clear_seeds(), ax_right.clear(), ax_right.imshow(img_raw), fig.canvas.draw_idle()])

plt.show()