import math

class dichotomy_method:
    def __init__(self, func, a, b, eps=1e-4, max_iter=1000):
        self.f = func
        self.a = a
        self.b = b
        self.eps = eps
        self.max_iter = max_iter
        self.root = None
        self.iterations = 0

    def solve(self, verbose=True):
        a, b = self.a, self.b
        fa, fb = self.f(a), self.f(b)

        if fa * fb > 0:
            raise ValueError(f"На відрізку [{a}, {b}] немає зміни знаку.")

        it = 0

        if verbose:
            print(f"{'Ітерація':>8} | {'a':>10} | {'b':>10} | {'c':>10} | {'f(c)':>12} | {'b - a':>10}")
            print("-" * 70)

        while (b - a) / 2 > self.eps and it < self.max_iter:
            c = (a + b) / 2
            fc = self.f(c)

            if verbose:
                print(f"{it:8d} | {a:10.6f} | {b:10.6f} | {c:10.6f} | {fc:12.6e} | {(b - a):10.6e}")

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            it += 1

        self.root = (a + b) / 2
        self.iterations = it

        if verbose:
            print("-" * 70)
            print(f"Знайдений корінь: x = {self.root:.8f}")
            print(f"Кількість ітерацій: {self.iterations}")
            print(f"Похибка: {(b - a) / 2:.2e}")

        return self.root

    def apriori_estimate(self):
        return math.ceil(math.log2((self.b - self.a) / self.eps))

    def __str__(self):
        return (f"Метод Дихотомії:\n"
                f"  Інтервал: [{self.a:.4f}, {self.b:.4f}]\n"
                f"  Корінь: {self.root:.8f}\n"
                f"  f(root) = {self.f(self.root):.6e}\n"
                f"  Апріорна оцінка: {self.apriori_estimate()}\n")
