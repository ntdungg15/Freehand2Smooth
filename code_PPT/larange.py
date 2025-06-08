import math
import re


def parse_function(prompt):
    """
    Đọc biểu thức hàm f(x) từ input, trả về hàm f(x) sử dụng math module.
    """
    expr = input(prompt)
    # Tạo môi trường an toàn cho eval
    def f(x):
        return eval(expr, {"x": x, "math": math})
    return f


def parse_values(prompt):
    """
    Đọc chuỗi số, cho phép dấu phẩy thập phân bằng dấu phẩy hoặc dấu chấm,
    và phân tách theo khoảng trắng hoặc dấu phẩy hoặc chấm phẩy.
    Trả về danh sách float.
    """
    raw = input(prompt)
    # chuyển dấu phẩy thập phân giữa chữ số thành dấu chấm
    normalized = re.sub(r'(?<=\d),(?=\d)', '.', raw)
    tokens = re.split(r'[\s,;]+', normalized.strip())
    try:
        return [float(tok) for tok in tokens if tok]
    except ValueError:
        raise ValueError("Định dạng số không đúng. Vui lòng nhập lại danh sách số.")


def lagrange_interpolation(x_vals, y_vals, x_eval):
    """
    Tính nội suy Lagrange tại điểm x_eval.
    x_vals: danh sách các node x_i
    y_vals: danh sách f(x_i)
    """
    n = len(x_vals)
    result = 0.0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x_eval - x_vals[j]) / (x_vals[i] - x_vals[j])
        result += term
    return result


if __name__ == "__main__":
    print("=== Nội suy Lagrange tổng quát ===")
    # Nhập biểu thức hàm
    f = parse_function("Nhập biểu thức f(x), ví dụ 'math.cos(x)' hoặc 'math.sqrt(1+x)': ")
    # Nhập số node
    try:
        n = int(input("Nhập số điểm dữ liệu (n): "))
    except ValueError:
        print("Số điểm phải là số nguyên.")
        exit(1)

    # Nhập danh sách x_i
    x_vals = parse_values(f"Nhập {n} giá trị x (các node), phân tách bởi khoảng trắng/dấu phẩy: ")
    if len(x_vals) != n:
        print(f"Cần nhập đúng {n} giá trị x! Bạn đã nhập {len(x_vals)}.")
        exit(1)

    # Tính y_vals từ hàm f
    y_vals = [f(xi) for xi in x_vals]

    # Nhập điểm cần nội suy
    x_eval_list = parse_values("Nhập giá trị x cần nội suy (có thể nhập nhiều): ")
    for x_eval in x_eval_list:
        try:
            P = lagrange_interpolation(x_vals, y_vals, x_eval)
            print(f"P({x_eval}) = {P:.10f}")
        except Exception as e:
            print("Lỗi khi tính nội suy cho x =", x_eval, ":", e)
