import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    # Tham số hóa theo t (arc-length)
    points = np.column_stack((x, y))
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    t = np.concatenate(([0], np.cumsum(dists)))
    if t[-1] == 0:
        t = np.linspace(0, 1, len(x))
    else:
        t = t / t[-1]
    t_dense = np.linspace(0, 1, 1000)
    # Spline bậc 3 2D
    if len(x) >= 4:
        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)
        x_spline = spline_x(t_dense)
        y_spline = spline_y(t_dense)
    else:
        x_spline = np.interp(t_dense, t, x)
        y_spline = np.interp(t_dense, t, y)
    # Bình phương tối thiểu bậc 5 2D
    if len(x) >= 6:
        poly_x = np.poly1d(np.polyfit(t, x, deg=5))
        poly_y = np.poly1d(np.polyfit(t, y, deg=5))
        x_poly = poly_x(t_dense)
        y_poly = poly_y(t_dense)
    elif len(x) >= 2:
        deg = min(3, len(x)-1)
        poly_x = np.poly1d(np.polyfit(t, x, deg=deg))
        poly_y = np.poly1d(np.polyfit(t, y, deg=deg))
        x_poly = poly_x(t_dense)
        y_poly = poly_y(t_dense)
    else:
        x_poly = x
        y_poly = y
    # Lưu CSV
    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame({
        'x_spline': x_spline,
        'y_spline': y_spline,
        'x_bptt': x_poly,
        'y_bptt': y_poly
    })
    df.to_csv("../data/smoothing_data.csv", index=False)
    # Vẽ kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r.', label='Gốc (thô)', alpha=0.5)
    plt.plot(x_spline, y_spline, 'b-', label='Spline bậc 3 (2D)', linewidth=2)
    plt.plot(x_poly, y_poly, 'g--', label='BPTT bậc 5 (2D)', linewidth=2)
    plt.title("📈 Làm mượt đường vẽ tay (Spline 2D & BPTT)", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/output.png", dpi=300)
    plt.show()
    print("✅ Đã lưu ảnh: output/output.png")
    print("✅ Đã xuất CSV: ../data/smoothing_data.csv")

if __name__ == "__main__":
    main()
