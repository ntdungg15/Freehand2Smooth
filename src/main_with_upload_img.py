import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Button, Label
from scipy.interpolate import splprep, splev
import os

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def extract_and_smooth_path(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None, None

    # Chọn contour dài nhất
    contour = max(contours, key=len)
    contour = contour[:, 0, :]  # bỏ chiều thừa

    # Chuyển về tọa độ x, y
    x = contour[:, 0]
    y = contour[:, 1]

    # Dùng spline nội suy mượt hơn
    tck, u = splprep([x, y], s=100000)  # tăng s để mượt hơn
    unew = np.linspace(0, 1.0, num=100000)
    out = splev(unew, tck)

    # Lật y để khớp với hệ tọa độ ảnh gốc (vì matplotlib gốc là tọa độ toán học, y tăng lên trên)
    img_height = img.shape[0]
    y = img_height - y
    out_y_flipped = img_height - np.array(out[1])

    return (x, y), (out[0], out_y_flipped), img_height

def plot_result(original, smoothed):
    fig, ax = plt.subplots()
    ax.plot(original[0], original[1], 'gray', linewidth=1, label="Original")
    ax.plot(smoothed[0], smoothed[1], 'red', linewidth=2, label="Smoothed")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "smoothed_result.png")
    plt.savefig(output_path)
    plt.show()
    print(f"\u0110\u00e3 l\u01b0u \u1ea3nh k\u1ebf t\u1ea3 qu\u1ea3 t\u1ea1i: {output_path}")

def save_to_csv(smoothed):
    csv_path = os.path.join(output_dir, "smoothed_path.csv")
    np.savetxt(csv_path, np.column_stack(smoothed), delimiter=",", header="x,y", comments='')
    print(f"\u0110\u00e3 l\u01b0u file CSV t\u1ea1i: {csv_path}")

def run():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    original, smoothed, _ = extract_and_smooth_path(file_path)
    if original is None:
        print("Kh\u00f4ng t\u00ecm th\u1ea5y \u0111\u01b0\u1eddng trong \u1ea3nh!")
        return

    plot_result(original, smoothed)
    save_to_csv(smoothed)

def main():
    root = Tk()
    root.title("Smooth Line Extractor")
    root.geometry("300x100")

    Label(root, text="Ch\u1ecdn \u1ea3nh ch\u1ee9a \u0111\u01b0\u1eddng v\u1ebd:").pack(pady=10)
    Button(root, text="Ch\u1ecdn \u1ea3nh", command=run).pack()

    root.mainloop()

if __name__ == '__main__':
    main()