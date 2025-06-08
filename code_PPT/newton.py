import re


def parse_input_list(prompt):
    """
    Đọc chuỗi đầu vào, xử lý dấu phẩy thập phân và phân tách thành danh sách float.
    Cho phép dấu phẩy decimal và ký tự phân tách là khoảng trắng hoặc dấu phẩy.
    """
    raw = input(prompt)
    # chuyển dấu phẩy thập phân giữa số thành dấu chấm
    normalized = re.sub(r'(?<=\d),(?=\d)', '.', raw)
    tokens = re.split(r'[\s,]+', normalized.strip())
    try:
        return [float(tok) for tok in tokens if tok]
    except ValueError:
        raise ValueError("Định dạng số không đúng. Vui lòng nhập lại.")


def divided_difference(x, y):
    """
    Tính hệ số cho đa thức Newton phân sai.
    Trả về danh sách coef sao cho:
      P(x) = coef[0] \
             + coef[1]*(x - x0) \
             + coef[2]*(x - x0)*(x - x1) \
             + ...
    """
    n = len(x)
    coef = y.copy()
    # Tính hệ số phân sai từ cấp 1 đến cấp n-1
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    return coef


def newton_polynomial(x, coef, x_eval):
    """
    Tính giá trị nội suy tại x_eval theo đa thức Newton phân sai.
    """
    n = len(x)
    result = coef[n-1]
    for i in range(n-2, -1, -1):
        result = result * (x_eval - x[i]) + coef[i]
    return result


if __name__ == "__main__":
    print("=== Nội suy Newton phân sai (tổng quát) ===")
    try:
        n = int(input("Nhập số điểm dữ liệu n: "))
    except ValueError:
        print("n phải là số nguyên.")
        exit(1)

    xs = parse_input_list(
        "Nhập các giá trị x (phân tách bởi dấu cách hoặc dấu phẩy; decimal có thể dùng dấu phẩy): "
    )
    ys = parse_input_list(
        "Nhập các giá trị y tương ứng (phân tách bởi dấu cách hoặc dấu phẩy; decimal có thể dùng dấu phẩy): "
    )

    if len(xs) != n or len(ys) != n:
        print("Lỗi: số lượng giá trị x và y phải bằng n.")
        exit(1)

    x_eval_list = parse_input_list("Nhập giá trị x cần nội suy: ")
    if not x_eval_list:
        print("Vui lòng nhập giá trị x cần nội suy.")
        exit(1)
    x_eval = x_eval_list[0]

    try:
        coef = divided_difference(xs, ys)
        result = newton_polynomial(xs, coef, x_eval)
        print(f"P({x_eval}) = {result:.10f}")
    except Exception as e:
        print("Có lỗi khi tính nội suy:", e)
