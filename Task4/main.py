import math as m
import numpy as np
import matplotlib.pyplot as plot
from scipy import integrate

class Func:
    def __init__(self):
        self.x = 0.0
        self.m = 1

    def set_x(self, x):
        self.x = x

    def set_m(self, m):
        self.m = m

    def func(self, t):
        return np.cos(self.m * t - self.x * np.sin(t))

# интервал по x
x_l = 0
x_h = 2 * np.pi

t_l = 0 #нижний предел интегрирования
t_h = np.pi #верхний предел


def integrate_Simpson(f, N, a, b):
    h = (b - a) / N
    # print("h:", h)
    S = 0

    f_xi = f(a)  # f(x_i)
    f_x2 = 0  # f((x_i + x_i+1)/2) - значение функции в середине отрезка
    f_xi1 = 0  # f(x_i+1)

    for i in range(1, N + 1):
        f_x2 = f(a + (i - 1 / 2) * h)
        f_xi1 = f(a + i * h)
        S += f_xi + 4 * f_x2 + f_xi1
        f_xi = f_xi1

    S = h * S
    return S


N = 30 #интервалы разбиения для интегрирования
k = 30 #интервалы для x
h_der = 1e-5 #шаг для вычисления производной методом конечной разности
x_step = (x_h - x_l)/k

x_s = [] #x_i
results = [] #значение разности в каждом x_i

x_i = x_l #x_0 = x_l = 0


F = Func()

for i in range(k + 1):
    x_i = x_l + i * x_step
    x_s.append(x_i)

    F.set_m(1)
    F.set_x(x_i)
    J1 = integrate.quadrature(F.func, t_l, t_h)[0] * (1 / np.pi)

    F.set_m(0)
    F.set_x(x_i + h_der)
    J0_right = integrate.quadrature(F.func,t_l, t_h)[0] * (1 / np.pi)

    F.set_x(x_i - h_der)
    J0_left = integrate.quadrature(F.func,t_l, t_h)[0] * (1 / np.pi)
    der = (J0_right - J0_left) / (2 * h_der)

    results.append(J1 + der)


plot.plot(x_s, results, 'ro')
plot.grid()
plot.xlabel('x_i')
plot.ylabel('J\'0 + J1')
plot.show()


#Сделать галку чтобы посмотреть шаг для вычисления производной#
k = 30  # интервалы для x

x_step = (x_h - x_l) / k

x_i = x_l  # x_0 = x_l = 0
F = Func()
results_h = []
h_val = []

h_der = 0.1

for j in range(10):
    h_der = h_der / 10
    h_val.append(h_der)
    res_max = 0

    for i in range(k + 1):
        x_i = x_l + i * x_step

        F.set_m(1)
        F.set_x(x_i)
        J1 = integrate.quadrature(F.func,t_l, t_h)[0] * (1 / np.pi)

        F.set_m(0)
        F.set_x(x_i + h_der)
        J0_right = integrate.quadrature(F.func,t_l, t_h)[0] * (1 / np.pi)

        F.set_x(x_i - h_der)
        J0_left = integrate.quadrature(F.func,t_l, t_h)[0] * (1 / np.pi)
        der = (J0_right - J0_left) / (2 * h_der)

        if abs(J1 + der) > res_max: #ищем максимальное по модулю значение
            res_max = abs(J1 + der)
    results_h.append(res_max)


plot.plot(h_val, results_h, 'bo')
plot.grid()
plot.xlabel('h_der')
plot.ylabel('max(|J\'0 + J1|)')
plot.loglog()
plot.title('    Зависимость максимального отклонения от h_der')
plot.show()
