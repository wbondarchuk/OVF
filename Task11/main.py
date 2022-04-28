import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

E0_sol = 1/2


def psi0_analytical(x):
    return (1/math.pi)**(1/4) * math.exp(-0.5*(x**2))


def U(x):
    return (x**2)/2


def make_matrix(N, x_start, x_end):
    h = (x_end - x_start) / N

    a = []
    b = []
    c = []

    for i in range(0, N):
        xi = x_start + i * h
        a.append(-0.5 / (h ** 2))
        b.append(1 / (h ** 2) + U(xi))
        c.append(-0.5 / (h ** 2))

    a[0] = 0
    c[N - 1] = 0

    A = [a, b, c]

    return A


def solve_diag(A, d, N):
    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(0, N):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]

    y = [i for i in range(N)]

    y[N - 1] = d_new[N - 1] / b_new[N - 1]

    for i in range(N - 2, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return y


x_start = -5
x_end = 5
N = 500
h = (x_end - x_start) / N

x_s_sol = []
psi0_sol = []

for i in range(N):
    x_s_sol.append(x_start + i * h)
    psi0_sol.append(psi0_analytical(x_s_sol[i]))

# нормировка
sol_norm = sla.norm(psi0_sol)
psi0_sol = [psi0_sol[i] / sol_norm for i in range(len(psi0_sol))]



x_start = -5
x_end = 5
N = 500
h = (x_end - x_start)/N

x_s = [(x_start + i*h) for i in range(N)]

psi0_sol = [psi0_analytical(x_s[i]) for i in range(N)]

sol_norm = sla.norm(psi0_sol)
psi0_sol = [psi0_sol[i]/sol_norm for i in range(len(psi0_sol))]

H = make_matrix(N, x_start, x_end) #трехдиагональная матрица

psi_next = [i/N for i in range(N)] #как начальный случайный вектор возьму просто N точек от 0 до 1

K = 50  # количество итераций

for k in range(0, K):
    psi_prev = psi_next
    psi_next = solve_diag(H, psi_prev, N)

E0 = sla.norm(psi_prev) / sla.norm(psi_next)

psi0_norm = sla.norm(psi_next)
psi0 = [psi_next[i] / psi0_norm for i in range(len(psi_next))]


plt.figure(figsize=(12, 7))
plt.subplot(121)
plt.plot(x_s_sol, psi0_sol, 'b+', label="psi_0(x)")
plt.grid()
plt.title('Волновая функция основного состояния, аналитическое решение')
plt.xlabel('x')
plt.ylabel('psi0(x)')
plt.legend()

plt.subplot(121)
plt.plot(x_s, psi0, 'r+', label="Численное решение")
plt.grid()
plt.title('Волновая функция основного состояния')
plt.xlabel('x')
plt.ylabel('psi0(x)')
plt.legend()
plt.show()


