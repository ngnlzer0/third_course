import math
from Dichotomy_method import dichotomy_method
from modified_Newton import ModifiedNewton


def f(x):
    return x * x + 5 * math.sin(x) - 1


def df(x):
    return 2 * x - 5 * math.cos(x)


def d2f(x):
    return 2 - 5 * math.sin(x)


def find_sign_change_intervals(f, xmin=-10, xmax=10, step=0.1):
    intervals = []
    x = xmin
    while x < xmax:
        a, b = x, x + step
        if f(a) * f(b) < 0:
            intervals.append((a, b))
        x += step
    return intervals


def choose_negative_interval(intervals):
    """Вибирає інтервал, який містить від’ємний корінь."""
    neg = [iv for iv in intervals if iv[0] < 0 or iv[1] < 0]
    if not neg:
        return None
    return min(neg, key=lambda iv: abs(iv[1]))


if __name__ == "__main__":
    eps = 1e-4
    intervals = find_sign_change_intervals(f)
    chosen = choose_negative_interval(intervals)

    if not chosen:
        print("Не знайдено від’ємних коренів у діапазоні.")
        exit(1)

    a, b = chosen
    x0 = (a + b) / 2

    print(f"Початковий інтервал: [{a:.4f}, {b:.4f}], x0 = {x0:.4f}, eps = {eps}\n")

    # --- Метод дихотомії ---
    bis = dichotomy_method(f, a, b, eps)
    bis.solve()
    print(bis)

    # --- Модифікований метод Ньютона ---
    newton = ModifiedNewton(f, df, d2f, -2.227)
    root = newton.solve(a, b)
    print(newton)

    # --- Порівняння ---
    print("\nПОРІВНЯННЯ РЕЗУЛЬТАТІВ:")
    diff = abs(bis.root - newton.root)
    print(f"  Різниця між коренями: {diff:.6e}")
