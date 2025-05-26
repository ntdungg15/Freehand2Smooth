import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Button, Label
# from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pandas as pd
import os

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def run():
    print("🖌️ VẼ TỰ DO BẰNG CHUỘT — DI CHUỘT ĐỂ VẼ ĐƯỜNG")
    
    image_path = filedialog.askopenfilename()
    if not image_path:
        return
    
    # drawer = FreehandDrawer()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return

    # Chọn contour dài nhất
    contour = max(contours, key=len)
    contour = contour[:, 0, :]  # bỏ chiều thừa

    # Chuyển về tọa độ x, y
    x = contour[:, 0]
    y = contour[:, 1]
    
    img_height = img.shape[0]
    y = img_height - y

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

def main():
    root = Tk()
    root.title("Smooth Line Extractor")
    root.geometry("300x100")

    Label(root, text="Ch\u1ecdn \u1ea3nh ch\u1ee9a \u0111\u01b0\u1eddng v\u1ebd:").pack(pady=10)
    Button(root, text="Ch\u1ecdn \u1ea3nh", command=run).pack()

    root.mainloop()

if __name__ == "__main__":
    main()
