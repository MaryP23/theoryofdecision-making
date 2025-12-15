from os import path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def f(x):
    x1, x2 = x
    return 2 * x1**2 + 3 * x2**2 - 2 * np.sin((x1 - x2) / 2) + x2
def g(x):
    x1, x2 = x
    return x1 + x2 - 1
def P(x):
    return g(x)**2
def F(x, k):
    return f(x) + k * P(x)
def hooke_jeeves(f_func, x0, lamd=0.1, alpha=2.0, eps=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    iter_count = 0
    for iter_count in range(max_iter):
        if lamd < eps:
            break
        # Все 8 направлений
        directions = [
            [lamd, 0], [-lamd, 0], [0, lamd], [0, -lamd],
            [lamd, lamd], [lamd, -lamd], [-lamd, lamd], [-lamd, -lamd]
        ]
        # Ищем лучшую точку среди всех 8
        best_point = x.copy()
        best_value = f_func(x)
        for dx, dy in directions:
            x_test = x.copy()
            x_test[0] += dx
            x_test[1] += dy
            value = f_func(x_test)
            if value < best_value:
                best_value = value
                best_point = x_test
        x_star = best_point
        # Поиск по образцу
        if best_value < f_func(x):
            x_pattern = x + alpha * (x_star - x)
            x = x_pattern if f_func(x_pattern) < best_value else x_star
        else:
            lamd /= 2
    return x, f_func(x), iter_count + 1
def penalty_method(x0):
    x = x0[:]
    k = 1.0
    s = 0
    while True:
        s += 1
        def Fk(x_local):
            return F(x_local, k)
        x, val, s = hooke_jeeves(Fk, x)
        if abs(g(x)) < 1e-6:
            break
        k *= 10
        if k > 100000000:
            break
    return x, f(x), s
x0 = [1.0, 0.0]
min_point, min_value, iter = penalty_method(x0)
print(f"Точка минимума: ({min_point[0]:.6f}, {min_point[1]:.6f})")
print(f"Минимальное значение функции: {min_value:.6f}")
print(f"Количество итераций: {iter}")
