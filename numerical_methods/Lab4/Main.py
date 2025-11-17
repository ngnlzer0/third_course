import numpy as np
import matplotlib.pyplot as plt
import math

X_LAGRANGE = [
    1, 8, 15, 22, 29,
    36, 43, 50, 57, 64,
    71, 78, 85, 92, 100
]

X_HERMITE_NODES = [
    1, 13, 25, 38, 50,
    62, 75, 87, 100
]

HERMITE_MULTIPLICITIES = [4, 4, 4, 4, 4, 4, 4, 4, 3]

def f(x): return np.log10(x)
def df1(x): return 1 / (x * np.log(10))
def df2(x): return -1 / (x ** 2 * np.log(10))
def df3(x): return 2 / (x ** 3 * np.log(10))

# 3. Метод Лагранжа

def lagrange_manual(x_target, nodes, values):
    n = len(nodes)
    L_val = 0.0
    for i in range(n):
        l_i = 1.0
        for j in range(n):
            if i != j:
                l_i *= (x_target - nodes[j]) / (nodes[i] - nodes[j])
        L_val += values[i] * l_i
    return L_val

# 4. Метод Ерміта

def build_hermite_table_explicit(unique_nodes, multiplicities):
    print(f"\n Початок побудови таблиці Ерміта...")

    # 1. Створення розширеного масиву
    x_rep = []
    for i in range(len(unique_nodes)):
        count = multiplicities[i]
        val = unique_nodes[i]
        for _ in range(count):
            x_rep.append(val)

    x_rep = np.array(x_rep)
    N = len(x_rep)

    print(f"Вузли з урахуванням кратності:\n {x_rep}")

    # 2. Ініціалізація таблиці
    table = np.zeros((N, N))

    # 3. Нульовий стовпець
    for i in range(N):
        table[i][0] = f(x_rep[i])

    # 4. Заповнення таблиці
    for j in range(1, N):
        for i in range(N - j):
            if x_rep[i] == x_rep[i + j]:
                # Похідні
                derivative = 0.0
                if j == 1:
                    derivative = df1(x_rep[i])
                elif j == 2:
                    derivative = df2(x_rep[i])
                elif j == 3:
                    derivative = df3(x_rep[i])

                table[i][j] = derivative / math.factorial(j)
            else:
                # Різниці
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_rep[i + j] - x_rep[i])

    print("Таблиця побудована успішно.")

    # Коефіцієнти Ньютона
    coeffs = table[0, :]

    # Вивід коефіцієнтів (перші 5 і останні 5, щоб не засмічувати консоль)
    print(f"Коефіцієнти полінома Ньютона (c0...c{N - 1}):")
    print(f"      Перші 5: {coeffs[:5]}")
    print(f"      ... ")
    print(f"      Останні 5: {coeffs[-5:]}")

    return x_rep, coeffs


def eval_hermite(x_val, nodes, coeffs):
    n = len(coeffs) - 1
    res = coeffs[n]
    for i in range(n - 1, -1, -1):
        res = res * (x_val - nodes[i]) + coeffs[i]
    return res

def format_arr(arr, precision=4):
    """Допоміжна функція: перетворює масив у красивий рядок чисел через кому"""
    # Якщо масив дуже великий, скорочуємо його
    if len(arr) > 10:
        start = ", ".join([f"{x:.{precision}f}" for x in arr[:4]])
        end = ", ".join([f"{x:.{precision}f}" for x in arr[-4:]])
        return f"[{start}, ..., {end}]"
    return "[" + ", ".join([f"{x:.{precision}f}" for x in arr]) + "]"


def main():
    np.set_printoptions(precision=4, suppress=True)

    # Точки для графіків та перевірки
    x_plot = np.linspace(1, 100, 500)
    y_true_arr = f(x_plot)

    # --- А) ЛАГРАНЖ ---
    print(f"\n{' МЕТОД ЛАГРАНЖА ':~^70}")  # ~~~ Заголовок ~~~
    print(f" • Кількість вузлів: {len(X_LAGRANGE)}")
    print(f" • Вузли (x): {format_arr(X_LAGRANGE)}")

    y_lag_nodes = [f(x) for x in X_LAGRANGE]
    print(f" • Значення (y): {format_arr(y_lag_nodes)}")

    # Обчислення масиву для графіку
    y_lag_arr = np.array([lagrange_manual(xi, X_LAGRANGE, y_lag_nodes) for xi in x_plot])

    # ---ЕРМІТ ---
    print(f"\n{' МЕТОД ЕРМІТА ':~^70}")

    h_nodes_rep, h_coeffs = build_hermite_table_explicit(X_HERMITE_NODES, HERMITE_MULTIPLICITIES)

    print(f" • Розмір полінома: {len(h_coeffs) - 1}-й ступінь")

    coeffs_str = format_arr(h_coeffs, precision=2)
    print(f" • Коефіцієнти Ньютона: {coeffs_str}")

    y_her_arr = np.array([eval_hermite(xi, h_nodes_rep, h_coeffs) for xi in x_plot])

    # --- В) АНАЛІЗ ТА ПОРІВНЯННЯ ---
    print(f"\n{' АНАЛІЗ РЕЗУЛЬТАТІВ ':~^70}")

    # 1. Контрольна точка
    x_test = 45
    y_exact = f(x_test)
    y_calc_lag = lagrange_manual(x_test, X_LAGRANGE, y_lag_nodes)
    y_calc_her = eval_hermite(x_test, h_nodes_rep, h_coeffs)

    print(f"\n>> КОНТРОЛЬНА ТОЧКА x = {x_test}")
    print("-" * 70)
    print(f"{'МЕТОД':<15} | {'ЗНАЧЕННЯ P(x)':<20} | {'ПОХИБКА |f-P|':<20}")
    print("-" * 70)
    print(f"{'Точне f(x)':<15} | {y_exact:<20.10f} | {'-':<20}")
    print(f"{'Лагранж':<15} | {y_calc_lag:<20.10f} | {abs(y_exact - y_calc_lag):<20.4e}")
    print(f"{'Ерміт':<15} | {y_calc_her:<20.10f} | {abs(y_exact - y_calc_her):<20.4e}")
    print("-" * 70)

    # 2. Статистика
    max_err_lag = np.max(np.abs(y_true_arr - y_lag_arr))
    max_err_her = np.max(np.abs(y_true_arr - y_her_arr))

    print(f"\n>> МАКСИМАЛЬНА ПОХИБКА НА ВІДРІЗКУ [1, 100]")
    print("-" * 70)
    print(f" • Max Err Лагранжа: {max_err_lag:.6e}")
    print(f" • Max Err Ерміта:   {max_err_her:.6e}")

    if max_err_her < max_err_lag:
        ratio = max_err_lag / max_err_her if max_err_her != 0 else 0
        print(f"\n[OK] Метод Ерміта точніший у {ratio:.1f} разів.")
    else:
        print("\n[!] Метод Ерміта показав більшу похибку (можливо через високий ступінь полінома/нестабільність).")

    print("=" * 70)
    print("Побудова графіків...")

    # --- ГРАФІКИ ---
    plt.figure(figsize=(14, 10))

    # Лагранж
    plt.subplot(2, 2, 1)
    plt.plot(x_plot, y_true_arr, 'k--', label='f(x)')
    plt.plot(x_plot, y_lag_arr, 'r-', label='L(x)')
    plt.scatter(X_LAGRANGE, y_lag_nodes, c='red', s=20, zorder=5)
    plt.title("Лагранж (15 точок)", fontsize=12)
    plt.ylim(-0.5, 2.5)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Ерміт
    plt.subplot(2, 2, 2)
    plt.plot(x_plot, y_true_arr, 'k--', label='f(x)')
    plt.plot(x_plot, y_her_arr, 'g-', label='H(x)')
    plt.scatter(X_HERMITE_NODES, [f(x) for x in X_HERMITE_NODES], c='green', s=30, zorder=5)
    plt.title(f"Ерміт ({len(h_coeffs) - 1}-й ступінь)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Похибка (Linear)
    plt.subplot(2, 2, 3)
    plt.plot(x_plot, np.abs(y_true_arr - y_lag_arr), 'r', label='Err Лагранж')
    plt.plot(x_plot, np.abs(y_true_arr - y_her_arr), 'g', label='Err Ерміт')
    plt.title("Абсолютна похибка (Linear)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Похибка (Log)
    plt.subplot(2, 2, 4)
    plt.semilogy(x_plot, np.abs(y_true_arr - y_lag_arr), 'r', label='Err Лагранж')
    plt.semilogy(x_plot, np.abs(y_true_arr - y_her_arr), 'g', label='Err Ерміт')
    plt.title("Абсолютна похибка (Log Scale)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("lab4_result.png")
    print("Графік збережено у файл lab4_result.png")
    plt.savefig("lab3")


if __name__ == "__main__":
    main()
