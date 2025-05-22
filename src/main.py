import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import os

def main():
    print("🎨 CHƯƠNG TRÌNH VẼ TAY & LÀM MƯỢT ĐƯỜNG")
    print("👉 Hướng dẫn:")
    print("  - Click chuột trái để vẽ đường cong (nhiều điểm)")
    print("  - Nhấn ENTER khi hoàn tất")
    print("  - Kết quả sẽ được vẽ và lưu ra file")

    # Bước 1: Vẽ và thu thập điểm
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("🖱️ Click chuột trái để vẽ tay — ENTER để hoàn tất", fontsize=12, color='blue')
    ax.set_xlabel("Trục X")
    ax.set_ylabel("Trục Y")
    ax.grid(True, linestyle='--', alpha=0.3)
    points = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(points) < 2:
        print("❌ Cần ít nhất 2 điểm để nội suy.")
        return

    # Bước 2: Tách x, y và sắp theo x tăng dần
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    # Nội suy spline và đa thức bậc 5 (BPTT)
    spline = CubicSpline(x, y)
    poly = np.poly1d(np.polyfit(x, y, deg=5))

    # Dữ liệu mịn để vẽ
    x_dense = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_dense)
    y_poly = poly(x_dense)

    # Bước 3: Vẽ kết quả
    # --- kiểm tra dữ liệu ---
    print("✔️ Số điểm gốc:", len(x))
    print("📈 Min/Max x:", np.min(x), np.max(x))
    print("📈 Min/Max y spline:", np.min(y_spline), np.max(y_spline))
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Dữ liệu gốc (vẽ tay)', markersize=6)
    plt.plot(x_dense, y_spline, 'b-', label='Spline nội suy', linewidth=2)
    plt.plot(x_dense, y_poly, 'g--', label='Bình phương tối thiểu (bậc 5)', linewidth=2)

    plt.title("📈 So sánh các phương pháp làm mượt đường vẽ tay", fontsize=14)
    plt.xlabel("Trục X")
    plt.ylabel("Trục Y")
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Tạo thư mục nếu chưa có
    os.makedirs("../data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Lưu kết quả
    df = pd.DataFrame({
        'x_dense': x_dense,
        'y_spline': y_spline,
        'y_bptt': y_poly
    })
    df.to_csv("../data/smoothing_data.csv", index=False)
    plt.savefig("output/output.png", dpi=300)

    # Hiển thị
    plt.show()

    # Thông báo
    print("✅ Đã lưu ảnh vào: output/output.png")
    print("✅ Đã xuất dữ liệu CSV vào: data/smoothing_data.csv")

if __name__ == "__main__":
    main()
