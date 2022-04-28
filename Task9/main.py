import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return math.sin(x)


def solution(x, x_0, x_n, y_0, y_n):
    c1 = (y_0 - y_n + math.sin(x_0) - math.sin(x_n)) / (x_0 - x_n)
    c2 = y_0 + math.sin(x_0) - c1 * x_0

    return -1 * math.sin(x) + c1 * x + c2


def system(n, x_0, x_n, y_0, y_n):
    h = (x_n - x_0) / n

    a = []  # элементы слева
    b = []  # элементы главной диагонали
    c = []  # элементы справа

    d = []  # вектор столбец справа
    x = []

    #0-ый элемент идет как база и дальше идет прогонка, потом обратный ход от n-ого элемента, поэтому нужна эта база
    # h^2*y(x0)[const]=y(x0)
    a.append(0)
    b.append(1)
    c.append(0)

    for i in range(1, n):
        a.append(1)
        b.append(-2)
        c.append(1)

    a.append(0)
    b.append(1)
    c.append(0)

    d.append(y_0)
    x.append(x_0)

    for i in range(1, n):
        x_i = x_0 + h * i
        d.append(h * h * f(x_i))
        x.append(x_i)

    d.append(y_n)
    x.append(x_n)

    A = [a, b, c]

    return A, d, x


def solve_diag(A, d, n):
    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(1, n + 1):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]

    y = [i for i in range(n + 1)]

    y[n] = d_new[n] / b_new[n]

    for i in range(n - 1, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return y


y_0 = -2 * math.pi
y_n = 2 * math.pi

x_0 = 0
x_n = 10*math.pi

n = 200


A, d, x = system(n, x_0, x_n, y_0, y_n)
y = np.array(solve_diag(A, d, n))
y_sol = np.array([solution(x_i, x_0, x_n, y_0, y_n) for x_i in x])

plt.figure(figsize=(12, 7))
plt.subplot(121)
plt.plot(x, y, 'go', label="y(x)")
plt.plot(x, y_sol, 'r+', label="analytical")
plt.grid()
plt.xlabel('x')
plt.title('y(x)')
plt.legend()


plt.subplot(122)
plt.plot(x, np.abs(y_sol - y), 'r', label="error")
plt.grid()
plt.xlabel('x')
plt.title('y(x)')
plt.legend()
plt.show()
# посмотреть матрицу
