import math
import numpy
def f(x):
    x1, x2 = x
    return 2 * x1**2 + 3 * x2**2 - 2 * numpy.sin((x1 - x2) / 2) + x2
def g(x):
    x1, x2 = x
    return x1 + x2 - 1
def P(x):
    return g(x)**2
def F(x, k):
    return f(x) + k * P(x)
def hooke_jeeves(fun, x0, step=0.1, alpha=2.0, eps=1e-6):
    x = [x0[0], x0[1]]
    iter = 0
    while step > eps:
        iter += 1
        x_new = [x[0], x[1]]
        best_val = fun(x_new)
        points = [
            [x[0] + step, x[1]],
            [x[0] - step, x[1]],
            [x[0], x[1] + step],
            [x[0], x[1] - step],
            [x[0] + step, x[1] + step],
            [x[0] + step, x[1] - step],
            [x[0] - step, x[1] + step],
            [x[0] - step, x[1] - step]
        ]
        for p in points:
            val = fun(p)
            if val < best_val:
                best_val = val
                x_new = p
        if best_val >= fun(x):
            step /= 2
            continue
        x_pattern = [
            x_new[0] + alpha * (x_new[0] - x[0]),
            x_new[1] + alpha * (x_new[1] - x[1])
        ]
        if fun(x_pattern) < fun(x_new):
            x = x_pattern
        else:
            x = x_new

    return x, fun(x), iter
def penalty_method(x0):
    x = x0[:]
    k = 1.0
    it = 0
    while True:
        it += 1
        def Fk(x_local):
            return F(x_local, k)
        x, val, it = hooke_jeeves(Fk, x)
        if abs(g(x)) < 1e-6:
            break
        k *= 10
        if k > 100000000:
            break
    return x, f(x), it
x0 = [1.0, 0.0]
min_point, min_value, iter = penalty_method(x0)
print(f"Точка минимума: ({min_point[0]:.6f}, {min_point[1]:.6f})")
print(f"Минимальное значение функции: {min_value:.6f}")
print(f"Количество итераций: {iter}")