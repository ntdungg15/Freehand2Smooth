import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Button, Label, simpledialog
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def run(max_points):
    print("CHỌN ẢNH ĐỂ LÀM MƯỢT")
    
    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not image_path:
        return
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("Không tìm thấy đường vẽ!")
        return

    # Chọn contour dài nhất
    contour = max(contours, key=len)
    contour = contour[:, 0, :]  
    
    if contour.shape[0] > max_points:
        idx_ds = np.linspace(0, contour.shape[0] - 1, max_points, dtype=int)
        contour = contour[idx_ds]

    # Chuyển về tọa độ x, y
    x = contour[:, 0]
    y = contour[:, 1]
    
    img_height = img.shape[0]
    y = img_height - y

    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    polyorder = 3
    max_window = min(51, len(x))
    if max_window % 2 == 0:
        max_window -= 1
    window = max_window

    if window >= polyorder + 2:
        try:
            x_smooth = savgol_filter(x, window_length=window, polyorder=polyorder)
            y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
        except ValueError:
            # Nếu vẫn lỗi, fallback về dữ liệu gốc
            x_smooth, y_smooth = x, y
    else:
        x_smooth, y_smooth = x, y
    
    spline = CubicSpline(x_smooth, y_smooth)
    x_dense = np.linspace(min(x_smooth), max(x_smooth), 1000)
    y_spline = spline(x_dense)

    poly = np.poly1d(np.polyfit(x_smooth, y_smooth, deg=5))
    y_poly = poly(x_dense)

    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    df = pd.DataFrame({
        'x_dense': x_dense,
        'y_spline': y_spline,
        'y_bptt': y_poly
    })
    df.to_csv("../data/smoothing_data.csv", index=False)

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

def main():
    root = Tk()
    root.withdraw()
    
    max_points = simpledialog.askinteger(
        "Điểm mẫu tối đa",
        "Nhập số điểm gốc tối đa trong khoảng [10, 10000]:",
        initialvalue=500,
        minvalue=10,
        maxvalue=10000
    )
    if max_points is None:
        print("Bỏ qua xử lý: chưa nhập số điểm tối đa.")
        exit()
    
    run(max_points)

if __name__ == "__main__":
    main()
