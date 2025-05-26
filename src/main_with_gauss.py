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
        self.ax.set_title("Giữ chuột trái để vẽ tay — thả chuột để hoàn tất", fontsize=12, color='blue')
        self.ax.set_xlabel("Trục X")
        self.ax.set_ylabel("Trục Y")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        fig.canvas.mpl_connect('button_press_event',  self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('button_release_event',self.on_release)
        plt.show()

        # Trả về bản gốc theo thứ tự vẽ
        return np.array(self.xs), np.array(self.ys)


def main():
    print("VẼ TỰ DO BẰNG CHUỘT — DI CHUỘT ĐỂ VẼ ĐƯỜNG")
    drawer = FreehandDrawer()
    raw_x, raw_y = drawer.collect_points()

    if len(raw_x) < 2:
        print("❌ Cần ít nhất 2 điểm.")
        return

    # Giữ nguyên thứ tự vẽ cho đường gốc
    x = raw_x.copy()
    y = raw_y.copy()

    # Làm mượt nhẹ trước khi spline bằng Savitzky-Golay
    window = min(51, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if window >= 5:
        x_smooth = savgol_filter(x, window_length=window, polyorder=3)
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
    else:
        x_smooth = x
        y_smooth = y

    # Spline smoothing (ko bắt buộc đi qua từng điểm)
    smoothing_factor = len(x_smooth) * np.var(y_smooth) * 0.01
    spline = UnivariateSpline(x_smooth, y_smooth, s=smoothing_factor)
    x_dense = np.linspace(x_smooth.min(), x_smooth.max(), 1000)
    y_spline = spline(x_dense)

    # Least squares polynomial bậc 5
    y_poly = np.poly1d(np.polyfit(x_smooth, y_smooth, deg=min(5, len(x_smooth)-1)))(x_dense)

    # Gaussian smoothing trên spline
    y_gauss = gaussian_filter1d(y_spline, sigma=10)

    # Lưu CSV
    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame({
        'x_dense': x_dense,
        'y_spline': y_spline,
        'y_poly': y_poly,
        'y_gauss': y_gauss
    })
    df.to_csv("../data/smoothing_data.csv", index=False)

    # Vẽ so sánh
    plt.figure(figsize=(10, 6))
    plt.plot(raw_x, raw_y, 'r-', alpha=0.5, label='Gốc (theo thứ tự vẽ)')
    plt.plot(x_dense, y_spline, 'b-', linewidth=2, label='Spline Smoothing')
    plt.plot(x_dense, y_poly, 'g--', linewidth=2, label='Poly bậc 5 (LS)')
    plt.plot(x_dense, y_gauss, 'm-', linewidth=2, label='Gaussian σ=10')
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
