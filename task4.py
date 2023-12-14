import numpy as np

def objective_function(x):
    return x[0]**2 - 4*x[0] + x[1]**2 - 2*x[1]

def gradient(x):
    return np.array([2*x[0] - 4, 2*x[1] - 2])

def projection(x, lower_bound, upper_bound):
    return np.maximum(lower_bound, np.minimum(x, upper_bound))

def conditional_gradient_method(initial_x, tolerance, lower_bound, upper_bound):
    x_k = np.array(initial_x)
    
    while True:
        grad = gradient(x_k)
        
        # Найдем шаг по методу условного градиента
        alpha = 0.1
        x_k1 = projection(x_k - alpha * grad, lower_bound, upper_bound)
        
        # Проверка критерия остановки
        if np.linalg.norm(x_k1 - x_k) <= tolerance:
            break
        
        x_k = x_k1

    return x_k

# Начальное приближение
initial_x = [0, 0]

# Ограничения
lower_bound = [0, 0]
upper_bound = [1, 2]

# Точность
tolerance = 0.1

result = conditional_gradient_method(initial_x, tolerance, lower_bound, upper_bound)
print("Минимум функции:", objective_function(result))
print("Оптимальное решение:", result)
