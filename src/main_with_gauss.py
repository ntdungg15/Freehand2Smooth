import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os

class FreehandDrawer:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.drawing = False

    def on_press(self, event):
        if event.button == 1:
            self.xs = []
            self.ys = []
            self.drawing = True

    def on_motion(self, event):
        if self.drawing and event.xdata is not None and event.ydata is not None:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        if event.button == 1 and self.drawing:
            self.drawing = False
            plt.close()

    def collect_points(self):
        fig, self.ax = plt.subplots(figsize=(8, 6))
        # Bỏ emoji để tránh cảnh báo font
        self.ax.set_title("Giữ chuột trái để vẽ tay — thả chuột để hoàn tất", fontsize=12, color='blue')
        self.ax.set_xlabel("Trục X")
        self.ax.set_ylabel("Trục Y")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        fig.canvas.mpl_connect('button_press_event',  self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('button_release_event',self.on_release)
        plt.show()

        return np.array(self.xs), np.array(self.ys)

def main():
    print("VẼ TỰ DO BẰNG CHUỘT — DI CHUỘT ĐỂ VẼ ĐƯỜNG")
    drawer = FreehandDrawer()
    x, y = drawer.collect_points()

    if len(x) < 2:
        print("❌ Cần ít nhất 2 điểm.")
        return

    # Loại bỏ x trùng, giữ thứ tự tăng
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # (1) Tuỳ chọn smoothing spline (UnivariateSpline, không ép qua từng điểm)
    # Giá trị s: càng lớn càng mượt; thử tăng nếu cần
    smoothing_factor = len(x) * np.var(y) * 0.01
    spline = UnivariateSpline(x, y, s=smoothing_factor)

    # Tạo dãy dense để vẽ spline
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_spline = spline(x_dense)

    # (2) Tuỳ chọn Savitzky–Golay để làm mượt y gốc (nếu đủ điểm)
    if len(x) >= 5:
        window = min(101, len(x) if len(x)%2==1 else len(x)-1)
        y_sg = savgol_filter(y, window_length=window, polyorder=3)
    else:
        y_sg = y

    # Tạo dense cho SG (nội suy linear nếu cần)
    y_sg_dense = np.interp(x_dense, x, y_sg)

    # (3) Bình phương tối thiểu (Least Squares đa thức bậc 5) trên y gốc/smoothed
    poly = np.poly1d(np.polyfit(x, y_sg, deg=5))
    y_poly = poly(x_dense)

    # (4) Gaussian filter mạnh trên spline
    y_gauss = gaussian_filter1d(y_spline, sigma=10)  # tăng sigma để mượt hơn

    # Tạo thư mục lưu
    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Xuất CSV
    df = pd.DataFrame({
        'x_dense':  x_dense,
        'y_spline': y_spline,
        'y_sg':     y_sg_dense,
        'y_poly':   y_poly,
        'y_gauss':  y_gauss
    })
    df.to_csv("../data/smoothing_data.csv", index=False)

    # Vẽ toàn bộ so sánh
    plt.figure(figsize=(10, 6))
    plt.plot(x,           y,         'r--',  alpha=0.3, label='Gốc (thô)')
    plt.plot(x_dense,     y_spline,  'b-',  linewidth=2, label='Spline smoothing')
    # plt.plot(x_dense,     y_sg_dense,'c--', linewidth=2, label='Savitzky–Golay')
    plt.plot(x_dense,     y_poly,    'g-.', linewidth=2, label='Poly bậc 5 (LS)')
    plt.plot(x_dense,     y_gauss,   'm-',  linewidth=2, label='Gaussian σ=8')
    plt.title("So sánh các phương pháp làm mượt đường vẽ tay", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/output.png", dpi=300)
    plt.show()


    print("✅ Đã lưu ảnh: output/output.png")
    print("✅ Đã xuất CSV: ../data/smoothing_data.csv")

if __name__ == "__main__":
    main()
