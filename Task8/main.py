import math
import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy as np
import time

a = 998
b = 1998
c = -999
d = -1999


def f_u(t, u, v):
    return a * u + b * v


def f_v(t, u, v):
    return c * u + d * v


t_start = 0
t_end = 1

u_start = 0.00001
v_start = 1

N = 5000
h = (t_end - t_start) / N


def solution(t_start, t_end, u_start, v_start, N):
    alpha = (u_start + v_start) * math.exp(t_start)
    beta = -1 * (u_start + 2 * v_start) * math.exp(1000 * t_start)

    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N

    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)
        u_s.append(2 * alpha * math.exp(-1 * t_s[i]) + beta * math.exp(-1000 * t_s[i]))
        v_s.append(-1 * alpha * math.exp(-1 * t_s[i]) - beta * math.exp(-1000 * t_s[i]))

    return t_s, u_s, v_s


def euler_expl(f_1, f_2, t_start, t_end, u_start, v_start, N):
    start = time.time()

    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N

    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)
        u_s.append(u_s[i - 1] + h * f_1(t=t_s[i - 1], u=u_s[i - 1], v=v_s[i - 1]))
        v_s.append(v_s[i - 1] + h * f_2(t=t_s[i - 1], u=u_s[i - 1], v=v_s[i - 1]))

    finish = time.time()
    return t_s, u_s, v_s, (finish-start)


def euler_impl(f_1, f_2, t_start, t_end, u_start, v_start, N):
    start = time.time()

    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N

    A = np.array([[1 - h * a, -1 * h * b], [-1 * h * c, 1 - h * d]])

    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    prev = np.array([u_start, v_start])

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)
        cur = sla.inv(A).dot(prev)

        prev = cur

        u_s.append(cur[0])
        v_s.append(cur[1])

    finish = time.time()
    return t_s, u_s, v_s, (finish-start)


t_sol, u_sol, v_sol = solution(t_start, t_end, u_start, v_start, N)

plt.plot(t_sol, u_sol, 'b', label="u(t)")
plt.plot(t_sol, v_sol, 'r', label="v(t)")
plt.grid()
plt.xlabel('t')
plt.title('Аналитическое решения')
plt.legend()
# plt.show()

t_expl, u_expl, v_expl, time_expl = euler_expl(f_u, f_v, t_start, t_end, u_start, v_start, N)

print('Явный',time_expl)
plt.plot(t_expl, u_expl, 'b', label="u(t)")
plt.plot(t_expl, v_expl, 'r', label="v(t)")
plt.grid()
plt.xlabel('t')
plt.title('Явное решение')
plt.legend()
# plt.show()

t_impl, u_impl, v_impl, time_impl= euler_impl(f_u, f_v, t_start, t_end, u_start, v_start, N)

print('Неявный',time_impl)
plt.plot(t_impl, u_impl, 'b', label="u(t)")
plt.plot(t_impl, v_impl, 'r', label="v(t)")
plt.grid()
plt.xlabel('t')
plt.title('Неявное решение')
plt.legend()
# plt.show()


# зачем обратная матрица в неявном методе? скорости(время)