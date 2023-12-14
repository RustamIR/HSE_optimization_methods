import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    return 2 * x[0]**2 + (x[1] - 1)**2

def constraint(x):
    return 2 * x[0] + x[1]

# Штрафная функция
def penalty_function(x, mu):
    return objective_function(x) + mu * (constraint(x))**2

# Общая функция для оптимизации
def total_function(x, mu):
    return penalty_function(x, mu)

# Начальное приближение
initial_x = [0, 0]

# Параметр штрафа
mu = 1

result = minimize(total_function, initial_x, args=(mu,), constraints={'type': 'eq', 'fun': constraint})

print("Минимум функции:", result.fun)
print("Оптимальное решение:", result.x)
