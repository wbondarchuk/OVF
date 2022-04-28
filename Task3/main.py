import numpy as np
import matplotlib.pyplot as plot
from scipy.integrate import quad
import random


f1 = lambda x: 1 / (1 + x ** 2)
f2 = lambda x: x ** (1 / 3.0) * np.exp(np.sin(x))


def trapezoid(func, a, b, N):
    x = np.linspace(a, b, N + 1)
    y = np.vectorize(func)(x)
    draw(func, a, b, N, x, 'Trapezoid')
    return (b - a) / (2 * N) * np.sum(y[1:] + y[:-1])


def simpson(func, a, b, N):
    x = np.linspace(a, b, N + 1)
    y = np.vectorize(func)(x)
    draw(func, a, b, N, x, 'Simpson')
    return (b - a) / (3 * N) * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


color = ['magenta', 'red', 'green', 'cyan', 'pink', 'yellow']


def draw(func, a, b, N, x, name):
    col = random.choice(color)
    X = np.linspace(a, b, 100)
    Y = np.vectorize(func)(X)
    plot.plot(X, Y, color=col)
    for i in range(N):
        xs = [x[i], x[i], x[i + 1], x[i + 1]]
        ys = [0, func(x[i]), func(x[i + 1]), 0]
        plot.fill(xs, ys, edgecolor='grey', alpha=0.7)
    plot.title(f'{name}, N = {N}')
    plot.show()


def main(func, a, b, N):
    integral = quad(func, a, b)[0]
    for i in range(1, N):
        n = 2 ** i
        Trapezoid = trapezoid(func, a, b, n)
        err_t = abs(Trapezoid - integral)
        Simpson = simpson(func, a, b, n)
        err_s = abs(Simpson - integral)
        print(f'N: {n}\t|\t Trapezoid: {Trapezoid} \t|\t T_error: {err_t} \t|\t Simpson: {Simpson} \t|\t S_error: {err_s}')


main(f1, -1, 1, 6)
main(f2, 0.5, 1, 7)
#Вопрос про ошибку во второй функции, при каких условиях теор оценка не работает
