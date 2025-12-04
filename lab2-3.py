import numpy as np
def f(x):
    x1, x2 = x[0], x[1]
    return 2 * x1 ** 2 + 3 * x2 ** 2 - 2 * np.sin((x1 - x2) / 2) + x2
def gradient(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 4 * x1 - np.cos((x1 - x2) / 2)
    df_dx2 = 6 * x2 + np.cos((x1 - x2) / 2) + 1
    return np.array([df_dx1, df_dx2])
print("\n1. Градиентный спуск с оптимизацией шага:")
# Метод наискорейшего спуска с оптимальным шагом
def golden_section_search(phi_func, a=0, b=1, tol=1e-3, max_iter=100):
    golden_ratio = (5 ** 0.5 - 1) / 2  # ≈ 0.618
    x1 = b - golden_ratio * (b - a)
    x2 = a + golden_ratio * (b - a)
    f1 = phi_func(x1)
    f2 = phi_func(x2)

    for i in range(max_iter):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - golden_ratio * (b - a)
            f1 = phi_func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + golden_ratio * (b - a)
            f2 = phi_func(x2)
        if abs(b - a) < tol:
            break
    return (a + b) / 2
def steepest_descent_simple(f_func, grad_f, x0, epsilon=1e-4):
    x = np.array(x0, dtype=float)
    iter_count = 0
    trajectory = [x.copy()]
    while np.linalg.norm(grad_f(x)) > epsilon:
        iter_count += 1
        def phi(lmbda):
            return f_func(x - lmbda * grad_f(x))
        # Используем метод золотого сечения
        lmbda_opt = golden_section_search(phi, a=0, b=1, tol=1e-3)
        # Обновление точки
        x = x - lmbda_opt * grad_f(x)
        trajectory.append(x.copy())
        if iter_count > 1000:
            print("Превышено максимальное количество итераций!")
            break
    return x, f_func(x), iter_count, trajectory
# Тестирование градиентного спуска
x0 = [1.0, 1.0]
min_point_gd, min_value_gd, iter_gd, trajectory_gd = steepest_descent_simple(f, gradient, x0)
print("Минимум найден в точке:", min_point_gd)
print("Значение функции в минимуме:", min_value_gd)
print("Число итераций:", iter_gd)
print("\n2. Метод Хука-Дживса:")
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
# Тестирование метода Хука-Дживса
x0_hj = [1.0, 1.0]
min_point_hj, min_value_hj, iter_hj = hooke_jeeves(f, x0_hj)
print(f"Точка минимума: ({min_point_hj[0]:.6f}, {min_point_hj[1]:.6f})")
print(f"Минимальное значение функции: {min_value_hj:.6f}")
print(f"Количество итераций: {iter_hj}")

