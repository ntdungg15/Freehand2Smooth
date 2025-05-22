import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import pandas as pd
import os

class FreehandDrawer:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.drawing = False

    def on_press(self, event):
        if event.button == 1:  # Chuột trái
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
            plt.close()  # Đóng cửa sổ sau khi vẽ xong

    def collect_points(self):
        fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("🖱️ Giữ chuột trái để vẽ tay — thả chuột để hoàn tất", fontsize=12, color='blue')
        self.ax.set_xlabel("Trục X")
        self.ax.set_ylabel("Trục Y")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        self.line, = self.ax.plot([], [], 'r-', linewidth=2)

        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('button_release_event', self.on_release)

        plt.show()
        return np.array(self.xs), np.array(self.ys)

def main():
    print("🖌️ VẼ TỰ DO BẰNG CHUỘT — DI CHUỘT ĐỂ VẼ ĐƯỜNG")
    
    drawer = FreehandDrawer()
    x, y = drawer.collect_points()

    if len(x) < 2:
        print("❌ Cần ít nhất 2 điểm.")
        return

    # Lọc x trùng
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    # Làm mượt nhẹ bằng Savitzky-Golay (nhưng giữ nguyên số điểm)
    window = min(51, len(x) // 2 * 2 + 1)
    if window >= 5:
        x_smooth = savgol_filter(x, window_length=window, polyorder=3)
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
    else:
        x_smooth = x
        y_smooth = y

    # Nội suy spline (đi qua tất cả điểm mượt)
    spline = CubicSpline(x_smooth, y_smooth)
    x_dense = np.linspace(min(x_smooth), max(x_smooth), 1000)
    y_spline = spline(x_dense)

    # Bình phương tối thiểu
    poly = np.poly1d(np.polyfit(x_smooth, y_smooth, deg=5))
    y_poly = poly(x_dense)

    # Lưu CSV
    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    df = pd.DataFrame({
        'x_dense': x_dense,
        'y_spline': y_spline,
        'y_bptt': y_poly
    })
    df.to_csv("../data/smoothing_data.csv", index=False)

    # Vẽ kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r.', label='Gốc (thô)', alpha=0.3)
    plt.plot(x_dense, y_spline, 'b-', label='Spline nội suy', linewidth=2)
    plt.plot(x_dense, y_poly, 'g--', label='BPTT bậc 5', linewidth=2)
    plt.title("📈 Làm mượt đường vẽ tay (Spline đi qua mọi điểm)", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/output.png", dpi=300)
    plt.show()

    print("✅ Đã lưu ảnh: output/output.png")
    print("✅ Đã xuất CSV: ../data/smoothing_data.csv")

if __name__ == "__main__":
    main()
