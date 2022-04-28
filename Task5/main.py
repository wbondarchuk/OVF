from time import time

import numpy as np
import matplotlib.pyplot as plt
import math


def x(n):
    xk = []
    for k in range(0, n + 1, 1):
        xk.append(1 + float(k) / n)
    return xk


def y(xk):
    return [math.log(i) for i in xk]


def l(x, xk, i, n):
    l = 1
    for j in range(0, n + 1):
        if j == i:
            continue
        l *= (x - xk[j])
    return l


def Lagrange(x, xk, n):
    P = 0
    lx = []
    lk = []
    yk = y(xk)
    for i in range(0, n + 1):
        lx.append(l(x, xk, i, n))
    for i in range(0, n + 1):
        lk.append(l(xk[i], xk, i, n))
    for i in range(0, n + 1):
        P += yk[i] * lx[i] / lk[i]
    return P


def divided_difference(xk, k):
    Y = [xk, y(xk)]
    n = 0
    for i in range(2, k + 2):
        try:
            Y.append(
                [(Y[i - 1][j] - Y[i - 1][j + 1]) / (Y[0][j] - Y[0][j + 1 + n]) for j in range(0, len(Y[i - 1]) - 1)])
        except IndexError:
            n += 1
            continue
        n += 1
    return Y


def Newton(x, xk, n):
    Y = divided_difference(xk, n)
    P = Y[1][0]
    X = 1
    for k in range(1, n + 1):
        for j in range(0, k):
            X *= (x - xk[j])
        P += X * Y[k + 1][0]
        X = 1
    return P


n = 50
X = np.arange(1, 2.01, 0.01)
P_Lag = []
P_New = []
Ln = [math.log(i) for i in X]

start = time()
for i in range(4, n + 1):
    P_Lag.append([Lagrange(j, x(i), i) for j in X])


eps_L = []
for i in range(n - 3):
    eps_L.append([(P_Lag[i][j] - Ln[j]) for j in range(0, len(X))])
print(time()-start)

start = time()
eps_N = []
for i in range(4, n + 1):
    P_New.append([Newton(j, x(i), i) for j in X])


for i in range(n - 3):
    eps_N.append([(P_New[i][j] - Ln[j]) for j in range(0, len(X))])


print(time()-start)
eps = []
for j in range(n - 3):
    eps.append([abs(eps_N[j][i]) - abs(eps_L[j][i]) for i in range(len(eps_N[j]))])

fig, axes = plt.subplots(1, 2)
Xk = x(n)
Yk = y(Xk)
axes[0].plot(X, P_New[n - 4])
axes[0].plot(Xk, Yk, 'ro')
axes[1].plot(X, P_Lag[n - 4])
axes[1].plot(Xk, Yk, 'bo')
plt.show()


