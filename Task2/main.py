import math
import matplotlib.pyplot as plot
import sys
import numpy as np

a = 1e-12  # cm
U_0 = 50 * 1e6 * 1.6e-12  # 50 MeV -> erg (g * cm^2 / s^2)
m = 9.1e-28  # g
h = 1.054e-27  # erg*s


# FUNCTION
class Function:
    def __init__(self, a, U_0, m, h):
        self.a = a
        self.U_0 = U_0
        self.m = m
        self.h = h
        self.const = 2 * self.a * self.a * self.m * self.U_0 / (self.h * self.h)

    def f(self, y):
        return np.sqrt(1 / y - 1) - 1 / np.tan(np.sqrt(self.const * (1 - y)))

    def df(self, y):
        return -self.const / (2 * np.sqrt(1 - y) * (np.sin(self.const * np.sqrt(1 - y))) ** 2) - 1 / (
                2 * y * y * np.sqrt(1 / y - 1))


func = Function(a, U_0, m, h)
y1 = np.arange(0.01, 0.99, 0.01)
plot.plot(y1, func.f(y1))
plot.grid()
plot.xlabel('y')
plot.ylabel('f(y)')
plot.title('Function')
plot.show()

# DICHOTOMY
# начальная и требуемая точность
d_0 = 0.1
d = 0.000000001

# число итераций
N = round(np.log10(d_0 / d) / np.log10(2))
print(N)
# начальные границы отрезка
a = 0.1
b = 0.125
f_a = func.f(a)
f_b = func.f(b)

half = []

for i in range(N):
    mid = (a + b) / 2
    f_mid = func.f(mid)
    if f_a * f_mid <= 0:
        b = mid
        f_b = func.f(b)
    elif f_mid * f_b <= 0:
        a = mid
        f_a = f_mid
    half.append((a + b) / 2)
    if np.abs(f_a - 0) < d:
        root = a
        break
    if np.abs(f_b - 0) < d:
        root = b
        break

y2 = np.arange(0.1, 0.13, 0.005)
plot.plot(y2, func.f(y2), 'g', half, [0 for i in range(len(half))], 'ro')
plot.grid()
plot.xlabel('y')
plot.ylabel('f(y)')
plot.title('Dichotomy')
plot.show()

root_dichotomy = (a + b) / 2
print(func.df(root_dichotomy))

# SIMPLE ITER
lm = 1 / func.df(0.1)
print(lm)

y_n_prev = 0.0025/2  # n-1
y_n = 0.005/2  # n

y_iter = []
counter_iter = 0

while np.abs(y_n - y_n_prev) > d:
    last = y_n
    y_n = y_n_prev - lm * func.f(y_n_prev)
    y_n_prev = last
    y_iter.append(y_n)
    counter_iter += 1

print(counter_iter)
y3 = np.arange(0.1, 0.13, 0.005)
plot.plot(y3, func.f(y3), 'y', y_iter, [0 for i in range(len(y_iter))], 'ro')
plot.grid()
plot.xlabel('y')
plot.ylabel('f(y)')
plot.title('Simple Iterations')
plot.show()

# NEWTON
y_n_prev = 0.0025/2  # n-1
y_n = 0.005/2  # n

d = 0.00000001

y_Newton = []
counter_Newton = 0

while np.abs(y_n - y_n_prev) > d:
    last = y_n
    y_n = y_n_prev - func.f(y_n_prev) / func.df(y_n_prev)
    y_n_prev = last
    y_Newton.append(y_n)
    counter_Newton += 1

print(counter_Newton)
y4 = np.arange(0.1, 0.13, 0.005)
plot.plot(y4, func.f(y4), 'm', y_Newton, [0 for i in range(len(y_Newton))], 'ro')
plot.grid()
plot.xlabel('y')
plot.ylabel('f(y)')
plot.title('Newton')
plot.show()
