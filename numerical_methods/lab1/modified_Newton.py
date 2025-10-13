import math

class ModifiedNewton:
    def __init__(self, f, df, d2f, x0, eps=1e-6, max_iter=100):
        self.f = f
        self.df = df
        self.d2f = d2f
        self.x0 = x0
        self.eps = eps
        self.max_iter = max_iter
        self.history = []
        self.root = None
        self.iterations = 0
        self.q = None

    def check_conditions(self, a, b):
        """Перевірка умов збіжності та обчислення q0."""
        f0 = self.f(self.x0)
        d2f0 = self.d2f(self.x0)

        if f0 * d2f0 <= 0:
            print("Умова f(x0)*f''(x0) > 0 не виконується — метод може розбігтись.")

        # Обчислюємо M1, m1 на [a, b]
        xs = [a + i * (b - a) / 200 for i in range(201)]
        df_vals = [abs(self.df(x)) for x in xs if abs(self.df(x)) > 1e-12]
        if not df_vals:
            raise ValueError("На відрізку похідна близька до нуля — метод може не збігатись.")

        M1 = max(df_vals)
        m1 = min(df_vals)

        # q0
        self.q = (M1 - m1) / (M1 + m1)
        if self.q >= 1:
            print(f"q0 = {self.q:.4f} ≥ 1 → можлива розбіжність методу.")
        else:
            print(f"Умова збіжності виконується: q0 = {self.q:.4f} < 1")

    def solve(self, a, b, verbose=True):
        """Основний метод розв’язку з виводом ітерацій."""
        self.check_conditions(a, b)

        x = self.x0
        df_x0 = self.df(self.x0)  # фіксуємо похідну в початковій точці
        if abs(df_x0) < 1e-12:
            raise ZeroDivisionError("f'(x0) ≈ 0 — неможливо почати ітерації.")

        if verbose:
            print(f"{'Ітерація':>8} | {'x':>12} | {'f(x)':>12} | {'|Δx|':>12}")
            print("-" * 50)

        for i in range(1, self.max_iter + 1):
            fx = self.f(x)

            # Додаємо перевірки стабільності тут — усередині циклу
            if abs(fx) > 1e10 or abs(x) > 1e6:
                print(" Значення виходить за межі — зупиняю ітерації")
                break

            if abs(self.df(x)) < 1e-6:
                print(" Мала похідна, ризик розбігання — зупиняю")
                break

            x_new = x - fx / df_x0  # формула модифікованого Ньютона
            delta = abs(x_new - x)

            if verbose:
                print(f"{i:8d} | {x:12.8f} | {fx:12.6e} | {delta:12.6e}")

            self.history.append(x_new)

            if delta < self.eps:
                self.root = x_new
                self.iterations = i
                if verbose:
                    print("-" * 50)
                    print(f" Збіжність досягнута за {i} ітерацій")
                return x_new

            x = x_new

        print(" Досягнуто максимум ітерацій без збіжності.")
        self.root = x
        self.iterations = self.max_iter
        return x

    def apriori_estimate(self, x_star):
        """Апріорна оцінка кількості ітерацій."""
        if self.q is None or self.q <= 0 or self.q >= 1:
            return None
        try:
            n0 = math.ceil(math.log(abs(self.x0 - x_star) / self.eps) / math.log(1 / self.q)) + 1
            return n0
        except ValueError:
            return None

    def __str__(self):
        apriori = self.apriori_estimate(self.root) if self.root is not None else None
        return (f"Модифікований метод Ньютона:\n"
                f"  Початкове наближення: {self.x0:.6f}\n"
                f"  Корінь: {self.root:.8f}\n"
                f"  f(root) = {self.f(self.root):.6e}\n"
                f"  Ітерацій (фактичних): {self.iterations}\n"
                f"  q0 = {self.q:.6f}\n"
                f"  Апріорна оцінка: {apriori if apriori is not None else 'н/д'}\n")
