import re
import math


def parse_input_list(prompt):
    """
    Đọc chuỗi số, cho phép dấu decimal bằng dấu phẩy hoặc dấu chấm,
    phân tách bởi khoảng trắng, dấu phẩy hoặc chấm phẩy.
    Trả về danh sách float.
    """
    raw = input(prompt)
    normalized = re.sub(r'(?<=\d),(?=\d)', '.', raw)
    tokens = re.split(r'[\s,;]+', normalized.strip())
    try:
        return [float(tok) for tok in tokens if tok]
    except ValueError:
        raise ValueError("Định dạng số không đúng. Vui lòng nhập lại danh sách số.")


def gaussian_elimination(A, b):
    """
    Giải hệ tuyến tính Ax = b bằng phương pháp khử Gauss với pivoting.
    A: ma trận kích thước n x n, b: vector độ dài n.
    Trả về vector x.
    """
    n = len(b)
    # Gộp b vào A
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]
    for k in range(n):
        # pivot
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[max_row] = M[max_row], M[k]
        # eliminate
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
    # back substitution
    x = [0]*n
    for i in range(n-1, -1, -1):
        s = M[i][n]
        for j in range(i+1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def least_squares_poly(x_vals, y_vals, degree):
    """
    Tính đa thức bình phương tối thiểu bậc 'degree' gần đúng dữ liệu.
    Trả về list hệ số coeffs độ dài degree+1 sao cho
    P(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[degree]*x^degree.
    """
    n = len(x_vals)
    m = degree
    # Tính các tổng x^k
    S = [sum(x**k for x in x_vals) for k in range(2*m+1)]
    # Tính các tổng x^k * y
    T = [sum((x_vals[i]**k) * y_vals[i] for i in range(n)) for k in range(m+1)]
    # Ma trận hệ số (m+1)x(m+1)
    A = [[S[i+j] for j in range(m+1)] for i in range(m+1)]
    b = T[:]  # vector bên phải
    coeffs = gaussian_elimination(A, b)
    return coeffs


def evaluate_poly(coeffs, x):
    """
    Đánh giá đa thức tại điểm x.
    coeffs: list hệ số từ bậc 0 đến bậc m.
    """
    return sum(c * (x**i) for i, c in enumerate(coeffs))


if __name__ == "__main__":
    print("=== Least Squares Polynomial Fit ===")
    # Nhập dữ liệu
    n = int(input("Nhập số điểm dữ liệu n: "))
    x_vals = parse_input_list(f"Nhập {n} giá trị x, phân tách: ")
    y_vals = parse_input_list(f"Nhập {n} giá trị y, phân tách: ")
    if len(x_vals) != n or len(y_vals) != n:
        print("Số điểm không khớp.")
        exit(1)
    degree = int(input("Nhập bậc đa thức tối đa (degree): "))
    # Tính hệ số
    coeffs = least_squares_poly(x_vals, y_vals, degree)
    print("Hệ số đa thức bình phương tối thiểu:")
    for i, c in enumerate(coeffs):
        print(f"a_{i} = {c:.10f}")
    # Thêm đánh giá
    xs = parse_input_list("Nhập các giá trị x để đánh giá P(x): ")
    for xv in xs:
        y_fit = evaluate_poly(coeffs, xv)
        print(f"P({xv}) = {y_fit:.10f}")
