import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("🎨 CHƯƠNG TRÌNH VẼ TAY & LÀM MƯỢT ĐƯỜNG")
    print("Hướng dẫn:")
    print("  - Click chuột trái để vẽ đường cong (nhiều điểm)")
    print("  - Nhấn ENTER khi hoàn tất")
    print("  - Kết quả sẽ được vẽ và lưu ra file")

    # Bước 1: Vẽ và thu thập điểm
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title("🖱️ Click chuột trái để vẽ tay — ENTER để hoàn tất", fontsize=12, color='blue')
    ax.set_xlabel("Trục X")
    ax.set_ylabel("Trục Y")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-10, 10)  
    ax.set_ylim(-10, 10)
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

    # Nhập giá trị đạo hàm đầu và cuối cho spline ràng buộc
    try:
        d1 = float(input("🔧 Nhập đạo hàm đầu bên trái (x0): "))
        d2 = float(input("🔧 Nhập đạo hàm cuối bên phải (xn): "))
    except ValueError:
        print("❌ Vui lòng nhập số thực hợp lệ cho đạo hàm.")
        return

    # Nội suy spline và đa thức bậc 5 (BPTT)
    spline_natural = CubicSpline(x, y, bc_type='natural')
    spline_clamped = CubicSpline(x, y, bc_type=((1, d1), (1, d2)))
    poly = np.poly1d(np.polyfit(x, y, deg=5))

    # Dữ liệu mịn để vẽ
    x_dense = np.linspace(min(x), max(x), 500)
    y_natural = spline_natural(x_dense)
    y_clamped = spline_clamped(x_dense)
    y_poly = poly(x_dense)

    # Bước 3: Vẽ kết quả
    print("✔️ Số điểm gốc:", len(x))
    print("📈 Min/Max x:", np.min(x), np.max(x))
    print("📈 Min/Max y spline:", np.min(y_natural), np.max(y_natural))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Dữ liệu gốc (vẽ tay)', markersize=6)
    plt.plot(x_dense, y_natural, 'b-', label='Spline tự nhiên', linewidth=2)
    plt.plot(x_dense, y_clamped, 'm--', label=f'Spline ràng buộc (đạo hàm={d1:.2f}, {d2:.2f})', linewidth=2)
    plt.plot(x_dense, y_poly, 'g-.', label='Bình phương tối thiểu (bậc 5)', linewidth=2)

    plt.title("📈 So sánh các phương pháp làm mượt đường vẽ tay", fontsize=14)
    plt.xlabel("Trục X")
    plt.ylabel("Trục Y")
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Lưu kết quả
    df = pd.DataFrame({
        'x_dense': x_dense,
        'y_spline_natural': y_natural,
        'y_spline_clamped': y_clamped,
        'y_bptt': y_poly
    })
    df.to_csv("../data/smoothing_data.csv", index=False)
    plt.savefig("output/output.png", dpi=300)

    # Hiển thị
    plt.show()

    print("✅ Đã lưu ảnh vào: src/output/output.png")
    print("✅ Đã xuất dữ liệu CSV vào: data/smoothing_data.csv")

if __name__ == "__main__":
    main()
