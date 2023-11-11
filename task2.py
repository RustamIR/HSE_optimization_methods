# from scipy.optimize import minimize
# import numpy as np

# # Определение целевой функции
# def target_function(x):
#     x1, x2 = x
#     return x1**2 + np.exp(x2**2)

# # Определение градиента и гессиана целевой функции
# def gradient(x):
#     x1, x2 = x
#     grad_x1 = 2 * x1
#     grad_x2 = 2 * x2 * np.exp(x2**2)
#     return np.array([grad_x1, grad_x2])

# def hessian(x):
#     x2 = x[1]
#     hess_x2_x2 = 2 * np.exp(x2**2) + 4 * x2**2 * np.exp(x2**2)
#     return np.array([[2, 0], [0, hess_x2_x2]])

# # Начальное приближение
# initial_guess = [1, 1]

# # Вызов функции оптимизации (минимизации) с методом Ньютона
# result = minimize(target_function, initial_guess, method='Newton-CG', jac=gradient, hess=hessian,options={'disp': True, 'maxiter': 100})

# # Получение минимума функции и оптимального решения
# minimized_value = result.fun
# optimal_solution = result.x

# # Вывод результата
# print("Минимум функции:", minimized_value)
# print("Оптимальное решение:", optimal_solution)



# import numpy as np

# # Определение целевой функции
# def target_function(x):
#     x1, x2 = x
#     return x1**2 + np.exp(x2**2)

# # Определение градиента и гессиана целевой функции
# def gradient(x):
#     x1, x2 = x
#     grad_x1 = 2 * x1
#     grad_x2 = 2 * x2 * np.exp(x2**2)
#     return np.array([grad_x1, grad_x2])

# def hessian(x):
#     x2 = x[1]
#     hess_x2_x2 = 2 * np.exp(x2**2) + 4 * x2**2 * np.exp(x2**2)
#     return np.array([[2, 0], [0, hess_x2_x2]])

# # Начальное приближение
# x = np.array([1.0, 1.0])

# # Параметры метода Ньютона
# max_iterations = 100
# tolerance = 1e-6

# for iteration in range(max_iterations):
#     grad = gradient(x)
#     hess = hessian(x)

#     step = np.linalg.solve(hess, -grad)
#     x = x + step

#     if np.linalg.norm(grad) < tolerance:
#         break

# # Результат оптимизации
# minimized_value = target_function(x)
# optimal_solution = x

# print("Минимум функции:", minimized_value)
# print("Оптимальное решение:", optimal_solution)



import numpy as np
import matplotlib.pyplot as plt

# Определение целевой функции и градиента
def target_function(x):
    x1, x2 = x
    return x1**2 + 2 * x2**2

def gradient(x):
    x1, x2 = x
    grad_x1 = 2 * x1
    grad_x2 = 4 * x2
    return np.array([grad_x1, grad_x2])

# Начальное приближение
x = np.array([1.0, 1.0])

# Для визуализации процесса оптимизации
x_history = [x.copy()]  # Хранение истории точек для построения графика
f_history = [target_function(x)]  # Значения функции на каждой итерации

# Сопряженные направления
direction = -gradient(x)

# Градиентный метод с использованием сопряженных градиентов
max_iterations = 100
for _ in range(max_iterations):
    grad = gradient(x)

    # Вычисление шага оптимизации
    alpha = np.dot(direction, direction) / np.dot(gradient(x), direction)

    # Обновление текущей точки
    x = x + alpha * direction

    # Вычисление нового сопряженного направления
    beta = np.dot(gradient(x), gradient(x)) / np.dot(gradient(x_history[-1]), gradient(x_history[-1]))
    direction = -gradient(x) + beta * direction

    # Добавление текущей точки и значения функции в историю
    x_history.append(x.copy())
    f_history.append(target_function(x))

# Визуализация процесса оптимизации
x_history = np.array(x_history)
plt.figure(figsize=(8, 6))
plt.scatter(x_history[:, 0], x_history[:, 1], c='red', label='Точки оптимизации')
plt.plot(x_history[:, 0], x_history[:, 1], 'b--', alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Процесс оптимизации методом сопряженных градиентов')
plt.legend()
plt.show()

# Результат оптимизации
minimized_value = target_function(x)
optimal_solution = x
print("Минимум функции:", minimized_value)
print("Оптимальное решение:", optimal_solution)

