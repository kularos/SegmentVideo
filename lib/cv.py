import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.colors import to_rgba

FEATURE_COLORS = ['deepskyblue', 'lime', 'magenta', 'yellow', 'orange', 'cyan', 'pink']

# 1. Load Image
img_path = "test_disps.png"
img_raw = plt.imread(f"../videos/{img_path}")
img_8bit = (img_raw[:, :, :3] * 255).astype(np.uint8) if img_raw.max() <= 1.0 else img_raw[:, :, :3].astype(np.uint8)
img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
H, W, _ = img_bgr.shape


# 2. Interactive Manager
class WatershedManager:
    def __init__(self, ax):
        self.ax, self.canvas = ax, ax.figure.canvas
        self.seeds = []
        self.current_id = 1
        self.active_seed = None
        self.last_res = None
        self.press_pos = None  # For drag vs click detection

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        # Check if clicking an existing seed
        for i, (patch, m_id) in enumerate(self.seeds):
            if patch.contains(event)[0]:
                self.active_seed = (patch, i)
                self.press_pos = (event.xdata, event.ydata)
                return

        # Add new seed if no seed was clicked
        if event.button == 1:
            color = FEATURE_COLORS[(self.current_id - 1) % len(FEATURE_COLORS)]
            new_patch = plt.Circle((event.xdata, event.ydata), radius=W // 80, color=color, alpha=0.7)
            self.ax.add_patch(new_patch)
            self.seeds.append((new_patch, self.current_id))
            self.canvas.draw()

    def on_motion(self, event):
        if self.active_seed is None or event.inaxes != self.ax: return
        patch, _ = self.active_seed
        patch.center = (event.xdata, event.ydata)
        self.canvas.draw()

    def on_release(self, event):
        if self.active_seed and self.press_pos:
            # If the mouse didn't move much, treat it as a deletion click
            dist = np.sqrt((event.xdata - self.press_pos[0]) ** 2 + (event.ydata - self.press_pos[1]) ** 2)
            if dist < W // 200:
                patch, idx = self.active_seed
                patch.remove()
                self.seeds.pop(idx)

        self.active_seed = None
        self.press_pos = None
        self.canvas.draw()

    def set_active_id(self, label):
        self.current_id = 1 if label == "Background" else int(label.split(" ")[1]) + 1


# 3. Layout Setup
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4)
ax_main = fig.add_subplot(gs[:, :3])
ax_debug_1, ax_debug_2 = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])
debug_axes = [ax_debug_1, ax_debug_2]

manager = WatershedManager(ax_main)
ax_main.imshow(img_raw)
ax_main.set_title("Click: Add | Drag: Move | Click Point: Remove")


def diagnose_ocr(event):
    if manager.last_res is None: return
    for ax in debug_axes: ax.clear(); ax.axis('off')
    features = [m for m in np.unique(manager.last_res) if m > 1]

    for i, m_id in enumerate(features[:2]):
        mask = (manager.last_res == m_id).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue

        # 1. Get the 4 corners of the mask itself
        c = max(cnts, key=cv2.contourArea)
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If we didn't get 4 points, fall back to the bounding box
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype(np.intp)

        # 2. Sort points specifically: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        rect_pts = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect_pts[0] = pts[np.argmin(s)]  # Top-Left
        rect_pts[2] = pts[np.argmax(s)]  # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect_pts[1] = pts[np.argmin(diff)]  # Top-Right
        rect_pts[3] = pts[np.argmax(diff)]  # Bottom-Left

        # 3. Calculate "True" Dimensions to eliminate parallelogram stretch
        (tl, tr, br, bl) = rect_pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 4. Correct for the 90deg orientation issue
        if maxHeight > maxWidth:
            # Shift mapping: TL->TR, TR->BR, BR->BL, BL->TL
            rect_pts = np.roll(rect_pts, 1, axis=0)
            maxWidth, maxHeight = maxHeight, maxWidth

        # 5. Perspective Warp to a clean RECTANGLE
        dst = np.float32([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]])

        M = cv2.getPerspectiveTransform(rect_pts, dst)
        warped = cv2.warpPerspective(img_8bit, M, (maxWidth, maxHeight))

        # Pre-process for OCR
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        debug_axes[i].imshow(thresh, cmap='gray')
        debug_axes[i].set_title(f"Corrected Feature {m_id - 1}")

    fig.canvas.draw_idle()


def run_watershed(event):
    if not manager.seeds: return
    markers = np.zeros((H, W), dtype=np.int32)
    markers[0, :], markers[-1, :], markers[:, 0], markers[:, -1] = 1, 1, 1, 1
    for (px, py), mid in [(s.center, mid) for s, mid in manager.seeds]:
        cv2.circle(markers, (int(px), int(py)), 3, mid, -1)
    manager.last_res = cv2.watershed(cv2.GaussianBlur(img_bgr, (5, 5), 0), markers)
    overlay = np.zeros((H, W, 4))
    for m_id in np.unique(manager.last_res):
        if m_id > 1:
            color = to_rgba(FEATURE_COLORS[(m_id - 1) % len(FEATURE_COLORS)])
            overlay[manager.last_res == m_id] = [*color[:3], 0.4]
    ax_main.clear();
    ax_main.imshow(img_raw);
    ax_main.imshow(overlay)
    fig.canvas.draw_idle()


# 5. UI Setup
ax_radio = plt.axes([0.02, 0.4, 0.08, 0.2])
radio = RadioButtons(ax_radio, ('Background', 'Feature 1', 'Feature 2'))
radio.on_clicked(manager.set_active_id)

btn_mask = Button(plt.axes([0.15, 0.02, 0.12, 0.05]), 'Update Mask', color='lime')
btn_mask.on_clicked(run_watershed)

btn_diag = Button(plt.axes([0.3, 0.02, 0.12, 0.05]), 'Diagnose', color='orange')
btn_diag.on_clicked(diagnose_ocr)

plt.show()