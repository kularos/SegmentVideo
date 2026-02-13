import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import to_rgba
import pytesseract

# --- CONFIGURATION ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

FEATURE_COLORS = ['deepskyblue', 'lime', 'magenta', 'yellow', 'orange', 'cyan', 'pink']

# 1. Load Image
img_path = "test_disps.png"
try:
    img_raw = plt.imread(f"../videos/{img_path}")
except:
    img_raw = np.zeros((500, 500, 3))  # Fallback
    cv2.putText(img_raw, "Img Not Found", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 2)

if img_raw.max() <= 1.0:
    img_8bit = (img_raw[:, :, :3] * 255).astype(np.uint8)
else:
    img_8bit = img_raw[:, :, :3].astype(np.uint8)
img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
H, W, _ = img_bgr.shape


# 2. Interactive Manager
class WatershedManager:
    def __init__(self, ax):
        self.ax, self.canvas = ax, ax.figure.canvas
        self.seeds = []
        self.current_id = 1
        self.btn_ref = None
        self.active_seed = None
        self.press = None
        self.last_res = None

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        for patch, m_id in self.seeds:
            if patch.contains(event)[0]:
                self.active_seed, self.press = patch, (patch.center, event.xdata, event.ydata)
                return
        if event.button == 1:
            color = FEATURE_COLORS[(self.current_id - 1) % len(FEATURE_COLORS)]
            new_patch = plt.Circle((event.xdata, event.ydata), radius=W // 60, color=color, alpha=0.7)
            self.ax.add_patch(new_patch);
            self.seeds.append((new_patch, self.current_id))
            self.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax: return
        (x0, y0), xpress, ypress = self.press
        self.active_seed.center = (x0 + event.xdata - xpress, y0 + event.ydata - ypress)
        self.canvas.draw()

    def on_release(self, event):
        self.active_seed = self.press = None

    def toggle_next_state(self, event):
        self.current_id += 1
        color = FEATURE_COLORS[(self.current_id - 1) % len(FEATURE_COLORS)]
        self.btn_ref.label.set_text(f"Adding: Feature {self.current_id - 1}")
        self.btn_ref.hovercolor = color
        self.canvas.draw()

    def clear(self):
        for p, _ in self.seeds: p.remove()
        self.seeds, self.current_id = [], 1
        self.btn_ref.label.set_text("Adding: Background")
        self.btn_ref.hovercolor = FEATURE_COLORS[0]
        self.canvas.draw()


# 3. Setup Layout (Two subplots now)
fig, (ax_left, ax_mid) = plt.subplots(1, 2, figsize=(15, 8))
plt.subplots_adjust(bottom=0.25)

manager = WatershedManager(ax_left)
ax_left.imshow(img_raw);
ax_left.set_title("1. Seed Entry")
ax_mid.imshow(img_raw);
ax_mid.set_title("2. Segmentation")


# 4. Logic Functions
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
            overlay[manager.last_res == m_id] = [*color[:3], 0.5]

    ax_mid.clear();
    ax_mid.imshow(img_raw);
    ax_mid.imshow(overlay)
    ax_mid.set_title("Segmentation Result")
    fig.canvas.draw_idle()


def read_displays(event):
    if manager.last_res is None: return
    # Clear previous text labels
    ax_mid.clear();
    ax_mid.imshow(img_raw)
    overlay = np.zeros((H, W, 4))

    for m_id in np.unique(manager.last_res):
        if m_id <= 1: continue

        # Draw the mask back on for the clear
        color_rgb = to_rgba(FEATURE_COLORS[(m_id - 1) % len(FEATURE_COLORS)])
        overlay[manager.last_res == m_id] = [*color_rgb[:3], 0.5]

        mask = (manager.last_res == m_id).astype(np.uint8)
        y, x = np.where(mask > 0)
        roi = img_8bit[y.min():y.max(), x.min():x.max()]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        try:
            val = pytesseract.image_to_string(thresh, config=config).strip()
            ax_mid.text(x.min(), y.min() - 10, f"Val: {val}", color='white',
                        bbox=dict(facecolor=FEATURE_COLORS[(m_id - 1) % len(FEATURE_COLORS)], alpha=0.8))
        except:
            print("OCR Error: Ensure Tesseract is installed.")

    ax_mid.imshow(overlay)
    fig.canvas.draw_idle()


# 5. UI
btn_state = Button(plt.axes([0.1, 0.05, 0.18, 0.07]), 'Adding: Background', color='lightgray',
                   hovercolor=FEATURE_COLORS[0])
manager.btn_ref = btn_state
btn_state.on_clicked(manager.toggle_next_state)

btn_calc = Button(plt.axes([0.3, 0.05, 0.15, 0.07]), 'Calculate', color='lime')
btn_calc.on_clicked(run_watershed)

btn_ocr = Button(plt.axes([0.47, 0.05, 0.15, 0.07]), 'Read Displays', color='orange')
btn_ocr.on_clicked(read_displays)

btn_clr = Button(plt.axes([0.64, 0.05, 0.12, 0.07]), 'Reset All')
btn_clr.on_clicked(lambda e: [manager.clear(), ax_mid.clear(), ax_mid.imshow(img_raw), fig.canvas.draw_idle()])

plt.show()