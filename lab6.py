import numpy as np

# Константа большого значения M
M = 1e6
# Ваша задача: F = 5x1 - 3x2 → min
# Ограничения:
# 1) 3x1 + 2x2 ≥ 6
# 2) 2x1 - 3x2 ≥ -6 → -2x1 + 3x2 ≤ 6
# 3) x1 - x2 ≤ 4
# 4) 4x1 + 7x2 ≤ 28
# x1, x2 ≥ 0
# Приведем к канонической форме:
# 1) 3x1 + 2x2 - x3 + x6 = 6  (x3 - избыточная, x6 - искусственная)
# 2) -2x1 + 3x2 + x4 = 6      (x4 - добавочная)
# 3) x1 - x2 + x5 = 4         (x5 - добавочная)
# 4) 4x1 + 7x2 + x7 = 28      (x7 - добавочная)
# Переменные: x1, x2, x3, x4, x5, x6, x7
# Целевая функция для M-метода: W = 5x1 - 3x2 + M*x6 → min
# Формирование начальной симплекс-таблицы
# Столбцы: [RHS, x1, x2, x3, x4, x5, x6, x7]
initial_table = np.array([
    # W: -5x1 + 3x2 - M*x6 = -6M (после обнуления коэффициента при x6)
    [-6 * M, -5 - 3 * M, 3 - 2 * M, M, 0, 0, 0, 0],  # Целевая функция W
    # x6: 3x1 + 2x2 - x3 + x6 = 6
    [6, 3, 2, -1, 0, 0, 1, 0],  # Искусственная переменная x6
    # x4: -2x1 + 3x2 + x4 = 6
    [6, -2, 3, 0, 1, 0, 0, 0],  # Добавочная переменная x4
    # x5: x1 - x2 + x5 = 4
    [4, 1, -1, 0, 0, 1, 0, 0],  # Добавочная переменная x5
    # x7: 4x1 + 7x2 + x7 = 28
    [28, 4, 7, 0, 0, 0, 0, 1]  # Добавочная переменная x7
], dtype=float)
def get_pivot_row(table, pivot_col_index):
    """
    Определяет индекс разрешающей строки для текущего разрешающего столбца.
    Выбирается строка с минимальным положительным отношением RHS/коэффициент.
    """
    ratios = []
    for i in range(1, len(table)):
        if table[i][pivot_col_index] > 1e-10:  # Только положительные коэффициенты
            ratio = table[i][0] / table[i][pivot_col_index]
            ratios.append((ratio, i))
        else:
            ratios.append((float('inf'), i))
    # Находим строку с минимальным положительным отношением
    min_ratio, min_row = min(ratios, key=lambda x: x[0])
    return min_row
def update_tableau(table, pivot_row_index, pivot_col_index):
    """
    Пересчитывает симплекс-таблицу после выбора разрешающего элемента.
    """
    pivot_element = table[pivot_row_index][pivot_col_index]
    # Создаем новую таблицу
    result_table = np.zeros_like(table, dtype=float)
    # 1. Нормируем разрешающую строку
    result_table[pivot_row_index] = table[pivot_row_index] / pivot_element
    # 2. Обновляем остальные строки
    for row_index in range(len(table)):
        if row_index == pivot_row_index:
            continue
        multiplier = table[row_index][pivot_col_index]
        result_table[row_index] = table[row_index] - multiplier * result_table[pivot_row_index]
    return result_table
def print_table(table, iteration, basis):
    """Выводит симплекс-таблицу на текущей итерации."""
    print(f"\n{'=' * 90}")
    print(f"ИТЕРАЦИЯ {iteration}")
    print(f"Базис: {basis}")
    print(f"{'-' * 90}")
    # Заголовок
    header = f"{'Базис':<8} | {'RHS':>10} |"
    var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    for name in var_names:
        header += f" {name:>10} |"
    print(header)
    print("-" * 90)
    # Строки таблицы
    for i in range(len(table)):
        if i == 0:
            row_name = 'W'
        else:
            row_name = basis[i - 1] if i - 1 < len(basis) else f'x{i}'
        row = f"{row_name:<8} | {table[i][0]:>10.4f} |"
        for j in range(1, len(table[i])):
            row += f" {table[i][j]:>10.4f} |"
        print(row)
    print("=" * 90)
# Основной цикл симплекс-метода
current_table = initial_table.copy()
iteration = 0
basis = ['x6', 'x4', 'x5', 'x7']  # Начальный базис
print("НАЧАЛЬНАЯ СИМПЛЕКС-ТАБЛИЦА")
print_table(current_table, iteration, basis)
# Флаг для отслеживания искусственной переменной
artificial_in_basis = True
while True:
    iteration += 1
    # Проверка оптимальности (для минимизации: все коэффициенты в целевой строке >= 0)
    # Проверяем только коэффициенты при переменных (столбцы 1..7)
    if np.all(current_table[0, 1:] >= -1e-10):
        print("\n✓ Все коэффициенты в целевой строке ≥ 0")
        # Проверяем, что искусственная переменная x6 равна 0
        if artificial_in_basis:
            # Находим строку с x6 в базисе
            for i, var in enumerate(basis):
                if var == 'x6':
                    if abs(current_table[i + 1][0]) < 1e-6:
                        print("✓ Искусственная переменная x6 = 0, решение допустимо")
                        artificial_in_basis = False
                    else:
                        print(f"✗ Искусственная переменная x6 = {current_table[i + 1][0]:.6f} ≠ 0")
                        print("Задача не имеет допустимых решений!")
                        break
            if not artificial_in_basis:
                print("✓ Оптимальное решение найдено!")
                break
        else:
            print("✓ Оптимальное решение найдено!")
            break
    # Выбор разрешающего столбца (минимальный коэффициент в целевой строке)
    # Исключаем столбец RHS (индекс 0)
    pivot_col_index = np.argmin(current_table[0, 1:]) + 1
    # Проверка неограниченности
    if np.all(current_table[1:, pivot_col_index] <= 1e-10):
        print("\n✗ Все коэффициенты в разрешающем столбце ≤ 0")
        print("Целевая функция неограниченна!")
        break
    # Выбор разрешающей строки
    try:
        pivot_row_index = get_pivot_row(current_table, pivot_col_index)
    except ValueError:
        print("Не удалось найти разрешающую строку")
        break
    print(f"\nРазрешающий столбец: x{pivot_col_index}")
    print(f"Разрешающая строка: {basis[pivot_row_index - 1]}")
    print(f"Разрешающий элемент: {current_table[pivot_row_index][pivot_col_index]:.4f}")
    # Обновляем базис
    var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    entering_var = var_names[pivot_col_index - 1]
    basis[pivot_row_index - 1] = entering_var
    # Если выводим искусственную переменную
    if entering_var == 'x6':
        artificial_in_basis = True
    elif 'x6' not in basis:
        artificial_in_basis = False
    # Пересчитываем таблицу
    current_table = update_tableau(current_table, pivot_row_index, pivot_col_index)
    # Выводим таблицу
    print_table(current_table, iteration, basis)
    if iteration > 20:
        print("Слишком много итераций!")
        break
# Вывод результатов
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ")
print("=" * 70)
# Извлекаем значения переменных
var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
solution = {}
for var in var_names:
    if var in basis:
        row_idx = basis.index(var)
        solution[var] = current_table[row_idx + 1][0]
    else:
        solution[var] = 0.0
print("\nОптимальные значения переменных:")
for i, var in enumerate(var_names):
    if i < 2 or abs(solution[var]) > 1e-6:  # Показываем x1, x2 и ненулевые переменные
        print(f"  {var} = {solution[var]:.6f}")
# Значение целевой функции (W в таблице, но нужно вычислить F)
F = 5 * solution['x1'] - 3 * solution['x2']
print(f"\nМинимальное значение целевой функции: Fmin = {F:.6f}")
print(f"Значение вспомогательной функции: W = {current_table[0][0]:.6f}")
# Проверка ограничений
print("\n" + "=" * 70)
print("ПРОВЕРКА ОГРАНИЧЕНИЙ")
print("=" * 70)
x1, x2 = solution['x1'], solution['x2']
# 1) 3x1 + 2x2 ≥ 6
check1 = 3 * x1 + 2 * x2
status1 = "✓" if check1 >= 6 - 1e-6 else "✗"
print(f"1) 3x1 + 2x2 = 3*{x1:.3f} + 2*{x2:.3f} = {check1:.3f} ≥ 6 {status1}")
# 2) 2x1 - 3x2 ≥ -6
check2 = 2 * x1 - 3 * x2
status2 = "✓" if check2 >= -6 - 1e-6 else "✗"
print(f"2) 2x1 - 3x2 = 2*{x1:.3f} - 3*{x2:.3f} = {check2:.3f} ≥ -6 {status2}")
# 3) x1 - x2 ≤ 4
check3 = x1 - x2
status3 = "✓" if check3 <= 4 + 1e-6 else "✗"
print(f"3) x1 - x2 = {x1:.3f} - {x2:.3f} = {check3:.3f} ≤ 4 {status3}")
# 4) 4x1 + 7x2 ≤ 28
check4 = 4 * x1 + 7 * x2
status4 = "✓" if check4 <= 28 + 1e-6 else "✗"
print(f"4) 4x1 + 7x2 = 4*{x1:.3f} + 7*{x2:.3f} = {check4:.3f} ≤ 28 {status4}")
# Проверка неотрицательности
print("\nПроверка неотрицательности переменных:")
for var in var_names:
    if var != 'x6' or solution[var] > 1e-6:  # x6 может быть неотрицательной, но если она искусственная, должна быть 0
        status = "✓" if solution[var] >= -1e-6 else "✗"
        print(f"  {var} = {solution[var]:.6f} ≥ 0 {status}")