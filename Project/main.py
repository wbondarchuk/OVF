import numpy as np
import matplotlib.pyplot as plt


# граничные условия
def top_gu(x):
    return 1, 0, 1 - (x / L) ** 2
    # return (1, 0, x/L)


def bot_gu(x):
    return 1, 0, 1 - (x / L) ** 2
    # return (1, 0, (x/L)-1)


def left_gu(y):
    return 1, 0, 1 - (y / L) ** 2
    # return (1, 0, 0.2*y/L)


def right_gu(y):
    return 1, 0, 1 - (y / L) ** 2
    # return (1, 0, 2-7*y/L)

# Решение всей системы по х
def Solution_x(L, ht, time, nt, n, U):
    xi = np.zeros(n + 1)
    eta = np.zeros(n + 1)
    D = np.zeros(n + 1)
    W = U * 0
    # т.к. у нас две переменные мы будем использовать шаг аналогичный в продольно поперечной схеме
    t = ht / 2
    h = 2 * L / n

    # Решение производится методом прогонки, надо задать значение диагоналей
    A = np.ones(n + 1) * (1 / (2 * h ** 2))
    B = np.ones(n + 1) * (1 / (h ** 2) + 2 / t)
    C = np.ones(n + 1) * (1 / (2 * h ** 2))

    for j in range(0, n + 1):
        # x,y,t
        [l1, l2, l3] = left_gu(-L + j * h)
        [r1, r2, r3] = right_gu(-L + j * h)
        xi[1] = -l2 / (l1 * h - l2)
        eta[1] = l3 * h / (l1 * h - l2)
        for i in range(1, n):
            Lxx = (U[i + 1, j] - 2 * U[i, j] + U[i - 1, j]) / (2 * h ** 2)#разностный оператор
            D[i] = -2 * U[i, j] / t - Lxx
            xi[i + 1] = C[i] / (B[i] - A[i] * xi[i])
            eta[i + 1] = (eta[i] * A[i] - D[i]) / (B[i] - A[i] * xi[i])

        W[n, j] = (r2 * eta[n] + r3 * h) / (r2 * (1 - xi[n]) + r1 * h)
        for i in range(n, 0, -1):
            W[i - 1, j] = W[i, j] * xi[i] + eta[i]

    return W

# Решение всей системы по у
def Solution_y(L, ht, time, nt, n, U):
    xi = np.zeros(n + 1)
    eta = np.zeros(n + 1)
    D = np.zeros(n + 1)
    W = U * 0
    # т.к. у нас две переменные мы будем использовать шаг аналогичный в продольно поперечной схеме
    t = ht / 2
    h = 2 * L / n

    # Решение производится методом прогонки, надо задать значение диагоналей
    A = np.ones(n + 1) * (1 / (2 * h ** 2))
    B = np.ones(n + 1) * (1 / (h ** 2) + 2 / t)
    C = np.ones(n + 1) * (1 / (2 * h ** 2))

    for i in range(0, n + 1):
        # x,y,t
        [b1, b2, b3] = bot_gu(-L + i * h)
        [t1, t2, t3] = top_gu(-L + i * h)
        xi[1] = -b2 / (b1 * h - b2)
        eta[1] = b3 * h / (b1 * h - b2)
        for j in range(1, n):
            Lxx = (U[i, j + 1] - 2 * U[i, j] + U[i, j - 1]) / (2 * h ** 2)
            D[j] = -2 * U[i, j] / t - Lxx
            xi[j + 1] = C[j] / (B[j] - A[j] * xi[j])
            eta[j + 1] = (eta[j] * A[j] - D[j]) / (B[j] - A[j] * xi[j])

        W[i, n] = (t2 * eta[n] + t3 * h) / (t2 * (1 - xi[n]) + t1 * h)
        for j in range(n, 0, -1):
            W[i, j - 1] = W[i, j] * xi[j] + eta[j]

    return W


def Diff_T(t, W1, W2):
    return (W2 - W1) / t - 1


# Задаём начальные данные

nt = 100
nx = 100
k = nx // 2
L = 1
T = 10
T_0 = 500
ht = T / nt #шаг по времени между слоями
h = 2 * L / nx #шаг по координате (равномерная сетка на плоскости)
Ta, time = [], []

x = np.linspace(-L, L, nx + 1)
y = np.linspace(-L, L, nx + 1)

X, Y = np.meshgrid(x, y)  # создаем сетку
# Надо задать всё пространство
W = np.zeros((nx + 1, nx + 1))
for i in range(nx + 1):
    for j in range(nx + 1):
        W[i][j] = T_0
print(W)
# Время
j = 0
top = h ** 2

while 1:

    temp = Solution_x(L, ht, ht * j, nt, nx, W)
    W = Solution_y(L, ht, ht * j, nt, nx, temp)

    Diff = np.amax(abs(Diff_T(ht / 2, temp, W)))-2

    Ta.append(W[k][k])
    time.append(ht * j)
    j += 1
    if Diff < top:
        break

print(W)
# Учёт границ
U = W

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.figure(figsize=(8, 6))
cs = plt.contourf(X, Y, U, 10)

plt.colorbar(cs)
plt.gca().set_aspect('equal') #одинаковое масштабирование

plt.figure()
plt.plot(time, Ta)
plt.xlabel('time')
plt.ylabel('T')
plt.show()
