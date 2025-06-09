import math
import re


def parse_input_list(prompt):
    """
    Đọc chuỗi số, cho phép dấu thập phân bằng dấu phẩy hoặc chấm,
    tách theo khoảng trắng hoặc dấu phẩy.
    """
    raw = input(prompt)
    normalized = re.sub(r'(?<=\d),(?=\d)', '.', raw)
    tokens = re.split(r'[\s,]+', normalized.strip())
    try:
        return [float(tok) for tok in tokens if tok]
    except:
        raise ValueError("Vui lòng nhập danh sách số hợp lệ.")


def cubic_spline_natural(x, y):
    """
    Tạo spline cubic natural: bậc ba với điều kiện bốn nhân.
    Trả về hệ số a, b, c, d của mỗi khoảng.
    """
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    # Tính alpha
    alpha = [0]*n
    for i in range(1, n-1):
        alpha[i] = 3*( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    # Ma trận tam giác l, mu, z
    l = [1] + [0]*(n-1)
    mu = [0]*n
    z = [0]*n
    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    l[n-1] = 1
    z[n-1] = 0
    c = [0]*n
    b = [0]*(n-1)
    d = [0]*(n-1)
    a = y[:-1]
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j]) /3
        d[j] = (c[j+1] - c[j]) / (3*h[j])
    return a, b, c[:-1], d


def cubic_spline_clamped(x, y, fp0, fpn):
    """
    Tạo spline cubic clamped (có điều kiện đạo hàm ở hai đầu).
    fp0 = f'(x0), fpn = f'(xn).
    """
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    alpha = [0]*n
    alpha[0] = 3*( (y[1]-y[0])/h[0] - fp0 )
    alpha[n-1] = 3*( fpn - (y[n-1]-y[n-2])/h[n-2] )
    for i in range(1, n-1):
        alpha[i] = 3*( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    l = [0]*n
    mu = [0]*n
    z = [0]*n
    l[0] = 2*h[0]
    mu[0] = 0.5
    z[0] = alpha[0] / l[0]
    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    l[n-1] = h[n-2]*(2 - mu[n-2])
    z[n-1] = (alpha[n-1] - h[n-2]*z[n-2]) / l[n-1]
    c = [0]*n
    b = [0]*(n-1)
    d = [0]*(n-1)
    a = y[:-1]
    c[n-1] = z[n-1]
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j]) /3
        d[j] = (c[j+1] - c[j]) / (3*h[j])
    return a, b, c[:-1], d


def evaluate_spline(x, coeffs, x_eval):
    """
    Đánh giá spline tại x_eval.
    coeffs = (a,b,c,d)
    """
    a, b, c, d = coeffs
    # tìm interval i sao cho x[i] <= x_eval <= x[i+1]
    for i in range(len(a)):
        if x_eval >= x[i] and x_eval <= x[i+1]:
            dx = x_eval - x[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    raise ValueError("x_eval ngoài khoảng của spline")

if __name__ == "__main__":
    print("=== Cubic Spline Interpolation ===")
    # nhập số điểm
    n = int(input("Nhập số điểm (n >= 2): "))
    # nhập x và y
    x = parse_input_list(f"Nhập {n} giá trị x tăng dần: ")
    y = parse_input_list(f"Nhập {n} giá trị f(x): ")
    if len(x)!=n or len(y)!=n:
        print("Số lượng điểm không hợp lệ.")
        exit(1)
    typ = input("Chọn loại spline ('natural' hoặc 'clamped'): ").strip().lower()
    if typ=='clamped':
        fp0 = float(input("Nhập f'(x0): "))
        fpn = float(input("Nhập f'(xn): "))
        coeffs = cubic_spline_clamped(x, y, fp0, fpn)
    else:
        coeffs = cubic_spline_natural(x, y)
    xs = parse_input_list("Nhập các giá trị x cần nội suy: ")
    for xv in xs:
        val = evaluate_spline(x, coeffs, xv)
        print(f"S({xv}) = {val:.10f}")
