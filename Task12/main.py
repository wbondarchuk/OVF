import math
import numpy as np
import matplotlib.pyplot as plt


def f(t):
    return a0 * math.sin(w0 * t) + a1 * math.sin(w1 * t)


def rec_w(k):
    if 0 <= k < n:
        return 1
    return 0


def hann_w(k):
    if 0 <= k < n:
        return 0.5 * (1 - math.cos(2 * math.pi * k / n))
    return 0


def no_w(k):
    return 1


def d_fourie(f, h, t_s):
    pow_spect = []
    w = []
    for i in range(n):
        fi = complex(0, 0)

        for k in range(n):
            fi += (1 / n) * f(t_s[k]) * np.exp(2 * np.pi * 1j * i * k / n) * h(k)

        pow_spect.append((fi * fi.conjugate()).real)  # мощность - квадрат модуля
        w.append(2 * np.pi * i / T)

    return w, pow_spect



# амплитуды
a0 = 1
a1 = 0.002

# частоты
w0 = 5.1
w1 = 5 * w0

# интервал
T = 2 * math.pi
t0 = 0
tn = T

# неправильная сетка
n = 50

# точки времени
t_s = np.linspace(t0, tn, n)
#
# plt.figure(figsize=(15, 6))
# plt.plot(t_s, [f(t) for t in t_s], 'bo-')
# plt.grid()
# plt.xlabel('t')
# plt.ylabel('f(t)')
# plt.title('Сигнал f(t)')
# plt.show()
#
# plt.subplot(121)
# plt.plot(np.arange(-50, n + 50), [rec_w(k) for k in np.arange(-50, n + 50)], 'r')
# plt.title("Прямоугольное oкно")
# plt.xlabel("k")
# plt.ylabel("h(k)")
#
# plt.subplot(122)
# plt.plot(np.arange(-50, n + 50), [hann_w(k) for k in np.arange(-50, n + 50)], 'b')
# plt.title("Окно Ханна")
# plt.xlabel("k")
# plt.ylabel("h(k)")
# plt.show()
#
# w_no, spec_no = d_fourie(f, no_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_no, spec_no, 'go-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощности, без окна')
#
# w_rec, spec_rec = d_fourie(f, rec_w, t_s)
#
# plt.plot(w_rec, spec_rec, 'ro-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощности, прямоугольное окно')
# # plt.show()
#
# w_hann, spec_hann = d_fourie(f, hann_w, t_s)
#
# plt.plot(w_hann, spec_hann, 'bo-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощности, окно Ханна')
# plt.show()
#
# w_no_2, spec_no_2 = d_fourie(f, no_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_no_2, spec_no_2, 'mo-')
# plt.grid()
# plt.xlabel('w')
# plt.yscale('log')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощность, без окна, лог. масштаб')
#
# plt.show()

# w_rec_2, spec_rec_2 = d_fourie(f, rec_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_rec_2, spec_rec_2, 'bo-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.yscale('log')
# plt.title('Спектр мощность, прямоугольное окно, лог. масштаб')
#
# plt.show()

w_hann_2, spec_hann_2 = d_fourie(f, hann_w, t_s)

plt.figure(figsize=(15, 6))
plt.plot(w_hann_2, spec_hann_2, 'bo-')
plt.grid()
plt.xlabel('w')
plt.ylabel('|f(w)|^2')
plt.yscale('log')
plt.title('Спектр мощность, окно Ханна, лог. масштаб')

plt.show()
