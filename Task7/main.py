import math
import matplotlib.pyplot as plt
import numpy as np

a = 10
b = 2
c = 2
d = 10

t_start = 0
t_end = 10

x_start = 5
y_start = 10


class F:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f_1(self, t, x, y):
        return self.a * x - self.b * x * y

    def f_2(self, t, x, y):
        return self.c * x * y - self.d * y


def rk2(f_1, f_2, t_start, t_end, x_start, y_start, N, alpha=0.75):
    x_s = []
    y_s = []
    t_s = []

    h = (t_end - t_start) / N

    t_s.append(t_start)
    x_s.append(x_start)
    y_s.append(y_start)

    x_shift = 0
    y_shift = 0
    t_shift = 0

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)

        f_1i = f_1(t=t_s[i - 1], x=x_s[i - 1], y=y_s[i - 1])
        f_2i = f_2(t=t_s[i - 1], x=x_s[i - 1], y=y_s[i - 1])

        x_shift = x_s[i - 1] + (h / (2 * alpha)) * f_1i
        y_shift = y_s[i - 1] + (h / (2 * alpha)) * f_2i
        t_shift = t_s[i - 1] + (h / (2 * alpha))

        x_s.append(x_s[i - 1] + h * ((1 - alpha) * f_1i + alpha * f_1(t=t_shift, x=x_shift, y=y_shift)))
        y_s.append(y_s[i - 1] + h * ((1 - alpha) * f_2i + alpha * f_2(t=t_shift, x=x_shift, y=y_shift)))

    return t_s, x_s, y_s


N = 150
func = F(a, b, c, d)

x_start = 5.1
y_start = 4.9

res_t, res_x, res_y = rk2(func.f_1, func.f_2, t_start, t_end, x_start, y_start, N)

# plt.plot(res_x, res_y, 'm.')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.title('Фазовая плоскость y(x)')
# plt.show()
#
# plt.plot(res_t, res_x, 'b', label='x(t)')
# plt.plot(res_t, res_y, 'r', label='y(t)')
# plt.grid()
# plt.xlabel('t')
# plt.title('Траектории x(t) и y(t)')
# plt.legend()
# plt.show()

# при каких условиях фазовая плоскость в точку(стационарные решения), и условие на h

x_st = d / c
y_st = a / b

x_start = x_st
y_start = y_st

func = F(a, b, c, d)

res_t, res_x, res_y = rk2(func.f_1, func.f_2, t_start, t_end, x_start, y_start, N)

plt.plot(res_x, res_y, 'm.')
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Фазовая плоскость y(x)')
plt.show()

plt.plot(res_t, res_x, 'b', label='x(t)')
plt.plot(res_t, res_y, 'r', label='y(t)')
plt.grid()
plt.xlabel('t')
plt.title('Траектории x(t) и y(t)')
plt.legend()
plt.show()

N_s = [i * 100 for i in range(1, 16)]
h_s = [(t_end - t_start) / n for n in N_s]
for i in range(len(N_s)):
    print(N_s[i], ':', h_s[i])
