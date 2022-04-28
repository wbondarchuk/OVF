import math
import numpy as np
import matplotlib.pyplot as plt


u_0t = 0 #u(0,t) = 0
u_Lt = 0 #u(L,t) = 0

def u_x0(x):
    return x*((1 - x/x_end)**2) #u(x,0) = x(1-x/L)^2

#разностный оператор
def Lu(j, m):
    return (u[m][j+1] - 2*u[m][j] + u[m][j-1]) / h**2

def make_matrix(n, x_start, x_end, tau):
    h = (x_start - x_end) / n

    a = []
    b = []
    c = []

    for i in range(0, n - 1):
        a.append(-0.5 * tau / h ** 2)
        b.append(1 + tau / h ** 2)
        c.append(-0.5 * tau / h ** 2)

    a[0] = 0
    c[n - 2] = 0

    A = [a, b, c]

    return A


def solve_diag(A, d, n):
    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(0, n - 1):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]

    y = [i for i in range(n - 1)]

    y[n - 2] = d_new[n - 2] / b_new[n - 2]

    for i in range(n - 3, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return y


x_start = 0
x_end = 2


#сетка по пространственной координате
N = 20
h = (x_end - x_start)/N

T = 20#точек по времени
t_start = 0
t_end = 10
tau = (t_end - t_start)/T  #шаг по времени

#значения в начальный момент времени t=0 во всех точках x_j

u = [[u_x0(x_start + j * h) for j in range(N + 1)]]
A = make_matrix(N, x_start, x_end, tau)

for m in range(T):

    # собирать правую часть
    d = []
    for j in range(1, N):
        d.append(u[m][j] + tau / 2 * Lu(j, m))

    # решение
    sol = solve_diag(A, d, N)
    # значения на концах
    sol.insert(0, 0)
    sol.append(0)

    u.append(sol)

u_max = []
t = []

for i in range(len(u)):
    u_max.append(max(u[i]))
    t.append(t_start+tau*i)

plt.figure(figsize=(10, 10))
plt.plot(t, u_max, 'b', label = 'u_max')
plt.grid()
plt.title('Зависимость максимальной температуры от времени')
plt.xlabel('t')
plt.ylabel('u_max')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(u, aspect='auto')
plt.grid()
plt.title('Распределение температуры по координате и времени')
plt.xlabel('x, номер точки')
plt.ylabel('t, номер точки')
plt.show()

# почему такое странное поведение и где экспонента?

