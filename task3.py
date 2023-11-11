from scipy.optimize import minimize

# Определение целевой функции
def target_function(x):
    x1, x2 = x
    return x1**2 + 2 * x2**2

# Начальное приближение
initial_guess = [1, 1]

# Вызов функции оптимизации (минимизации) с методом сопряженных градиентов
result = minimize(target_function, initial_guess, method='CG', options={'disp': True, 'maxiter': 100})

# Получение минимума функции и оптимального решения
minimized_value = result.fun
optimal_solution = result.x

# Вывод результата
print("Минимум функции:", minimized_value)
print("Оптимальное решение:", optimal_solution)


# import numpy as np

# # Определение целевой функции и её градиента
# def target_function(x):
#     x1, x2 = x
#     return x1**2 + 2 * x2**2

# def gradient(x):
#     x1, x2 = x
#     grad_x1 = 2 * x1
#     grad_x2 = 4 * x2
#     return np.array([grad_x1, grad_x2])

# # Начальное приближение
# x = np.array([1.0, 1.0])

# # Параметры метода сопряженных градиентов
# max_iterations = 100
# tolerance = 1e-6
# beta = 0.0  # Используется для обновления направления сопряженных градиентов

# # Начальное направление движения
# direction = -gradient(x)

# for iteration in range(max_iterations):
#     # Градиент текущей точки
#     grad_current = gradient(x)

#     if np.linalg.norm(grad_current) < tolerance:
#         break  # Критерий остановки: если градиент близок к нулю, завершаем оптимизацию

#     if iteration == 0:
#         direction = -grad_current
#     else:
#         beta = np.dot(grad_current, grad_current) / np.dot(grad_previous, grad_previous)
#         direction = -grad_current + beta * direction

#     # Шаг оптимизации (например, можно использовать фиксированный шаг или метод линейного поиска)
#     step_size = 0.1

#     # Обновление текущей точки
#     x = x + step_size * direction

#     grad_previous = grad_current

# # Результат оптимизации
# minimized_value = target_function(x)
# optimal_solution = x

# print("Минимум функции:", minimized_value)
# print("Оптимальное решение:", optimal_solution)
