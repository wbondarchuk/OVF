import numpy as np
import matplotlib.pyplot as plt
import math as m

n = 1000
t = np.linspace(0, 3, n)
x = np.zeros(n, dtype=float)
y1 = np.zeros(n, dtype=float)
y2 = np.zeros(n, dtype=float)
x[0] = 1


def y(x):
    y = m.exp(-x)
    return y


y1 = [y(x) for x in t]


def f(x):
    value = -x
    return value


# Euler's method
for i in range(0, n - 1):
    x[i + 1] = x[i] + (t[i + 1] - t[i]) * f(x[i])
for i in range(0, n):
    y2[i] = abs(y1[i] - x[i])


def plot():
    print('Euler method')
    plt.plot(t, x)
    plt.plot(t, y1)
    plt.title('Euler func')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    plt.plot(t, y2)
    plt.title('Euler dif')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()


# Runge2
xr2 = np.zeros(n, dtype=float)
yr22 = np.zeros(n, dtype=float)
xr2[0] = 1

for i in range(0, n - 1):
    ar2 = 3 / 4
    h = t[i + 1] - t[i]
    xr2[i + 1] = xr2[i] + h * ((1 - ar2) * f(xr2[i]) + ar2 * f(xr2[i] + h * f(xr2[i]) / (2 * ar2)))

for i in range(0, n):
    yr22[i] = abs(y1[i] - xr2[i])


def plotr2():
    print('Runge 2nd range')
    plt.plot(t, xr2)
    plt.plot(t, y1)
    plt.title('Runge 2')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    plt.plot(t, yr22)
    plt.title('Runge 2 dif')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()


# Runge4
xr4 = np.zeros(n, dtype=float)
yr42 = np.zeros(n, dtype=float)
xr4[0] = 1

for i in range(0, n - 1):
    hr4 = t[i + 1] - t[i]
    k1 = f(xr4[i])
    k2 = f(xr4[i] + hr4 * k1 / 2)
    k3 = f(xr4[i] + hr4 * k2 / 2)
    k4 = f(xr4[i] + hr4 * k3)
    xr4[i + 1] = xr4[i] + (hr4 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

for i in range(0, n):
    yr42[i] = abs(y1[i] - xr4[i])


def plotr4():
    print('Runge 4th range')
    plt.plot(t, xr4)
    plt.title('Runge 4')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    plt.plot(t, yr42)
    plt.title('Runge 4 dif')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

#
# plot()
# plotr2()
# plotr4()

ly2 = [m.log(x, 10) for x in y2[1:]]
lyr22 = [m.log(x, 10) for x in yr22[1:]]
lyr42 = [m.log(x, 10) for x in yr42[1:]]

plt.plot(t[1:], ly2, color='g', label='Euler')
plt.plot(t[1:], lyr22, color='b', label='Runge2')
plt.plot(t[1:], lyr42, color='r', label='Runge4')
plt.title('Dif')
plt.xlabel('t')
plt.ylabel('lg(x)')
plt.legend()
plt.show()

plt.plot(t, y1, color='black', label='exp')
plt.plot(t, x, color='g', label='Euler')
plt.plot(t, xr2, color='b', label='Runge2')
plt.plot(t, xr4, color='r', label='Runge4')
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.show()
