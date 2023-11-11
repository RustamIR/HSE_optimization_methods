import numpy as np
from scipy.optimize import minimize

# # Определение целевой функции и её градиента
def target_function(x):
    x1, x2 = x
    return 2 * x1**2 + x1 * x2 + 3 * x2**2

def target_gradient(x):
    x1, x2 = x
    grad_x1 = 4 * x1 + x2
    grad_x2 = x1 + 6 * x2
    return np.array([grad_x1, grad_x2])

# # Начальное приближение
initial_guess = [-1, 1]

# Параметры правила Армихо
alpha = 1  # Параметр альфа
beta = 1/2  # Параметр бета
epsilon = 1/2  # Параметр эпсилон

# # Вызов функции оптимизации с правилом Армихо
result = minimize(target_function, initial_guess, method='BFGS', jac=target_gradient,
                  options={'disp': True, 'maxiter': 100, 'gtol': epsilon,
                           'alpha': alpha, 'beta': beta})

# Вывод результата
print("Минимум функции:", result.fun)
print("Оптимальное решение:", result.x)



"""Второй способ """
# import numpy as np
# import matplotlib.pyplot as plt

# # Определение целевой функции и градиента
# def target_function(x):
#     x1, x2 = x
#     return x1**2 + x1*x2 + 3*x2**2

# def gradient(x):
#     x1, x2 = x
#     grad_x1 = 2 * x1 + x2
#     grad_x2 = x1 + 6 * x2
#     return np.array([grad_x1, grad_x2])

# # Начальное приближение
# x = np.array([-1.0, 1.0])

# # Для визуализации процесса оптимизации
# x_history = [x.copy()]  # Хранение истории точек для построения графика
# f_history = [target_function(x)]  # Значения функции на каждой итерации

# # Параметры правила Армихо
# alpha = 1.0
# epsilon = 0.5
# etta = 0.5

# # Градиентный метод с использованием правила Армихо
# max_iterations = 500
# for _ in range(max_iterations):
#     grad = gradient(x)

#     # Поиск шага оптимизации с использованием правила Армихо
#     t = 1.0
#     while target_function(x - t * grad) >= target_function(x) - epsilon * t * np.dot(grad, grad):
#         t *= etta

#     # Обновление текущей точки
#     x = x - t * grad

#     # Добавление текущей точки и значения функции в историю
#     x_history.append(x.copy())
#     f_history.append(target_function(x))

# # Визуализация процесса оптимизации
# x_history = np.array(x_history)
# plt.figure(figsize=(8, 6))
# plt.scatter(x_history[:, 0], x_history[:, 1], c='red', label='Точки оптимизации')
# plt.plot(x_history[:, 0], x_history[:, 1], 'b--', alpha=0.5)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Процесс оптимизации методом градиента с правилом Армихо')
# plt.legend()
# plt.show()

# # Результат оптимизации
# minimized_value = target_function(x)
# optimal_solution = x
# print("Минимум функции:", minimized_value)
# print("Оптимальное решение:", optimal_solution)
