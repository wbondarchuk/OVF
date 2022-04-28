# Машинным ε называется такое число, что 1 + ε/2 = 1, но 1 + ε 6= 1. (Также часто
# используется обозначение ULP – unit in the last place, или unit of least precision, единица
# в младшем разряде.) Найти машинное ε, число разрядов в мантиссе, максимальную
# и минимальную степени, при вычислениях с обычной и двойной точностью. Сравнить
# друг с другом четыре числа: 1, 1 + ε/2, 1 + ε и 1 + ε + ε/2, объяснить результат.
import numpy as np

print(np.finfo(np.float32))
print(np.finfo(np.float64))


def EpsFloat32(func_eps=np.float32):
    i = 0
    machine_epsilon = func_eps(1)
    while func_eps(1) + func_eps(machine_epsilon) != func_eps(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func_eps(machine_epsilon) / func_eps(2)
        i += 1
    return [machine_epsilon_last, i - 1]


def EpsFloat64(func_eps=np.float64):
    i = 0
    machine_epsilon = func_eps(1)
    while func_eps(1) + func_eps(machine_epsilon) != func_eps(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func_eps(machine_epsilon) / func_eps(2)
        i += 1
    return [machine_epsilon_last, i - 1]


ResultFloat32 = EpsFloat32()
ResultFloat64 = EpsFloat64()


def MaxPowerEpsFloat32():
    i = 0
    max_power = [np.float32(1.0)]
    while max_power[i] != np.inf:
        max_power.append(np.float32(max_power[i] * 2))
        i += 1
    return max_power[i - 1]


def MinPowerEpsFloat32():
    i = 0
    min_power = [np.float32(1.0)]
    while min_power[i] != np.float32(0.0):
        min_power.append(np.float32(min_power[i] / 2))
        i += 1
    return min_power[(i - 1 - ResultFloat32[1])]


def MaxPowerEpsFloat64():
    i = 0
    max_power = [np.float64(1.0)]
    while max_power[i] != np.inf:
        max_power.append(float(max_power[i] * 2))
        i += 1
    return max_power[i - 1]


def MinPowerEpsFloat64():
    i = 0
    min_power = [np.float64(1.0)]
    while min_power[i] != np.float64(0.0):
        min_power.append(np.float64(min_power[i] / 2))
        i += 1
    return min_power[(i - 1 - ResultFloat64[1])]


ResultFloat32Max = MaxPowerEpsFloat32()
ResultFloat32Min = MinPowerEpsFloat32()
ResultFloat64Max = MaxPowerEpsFloat64()
ResultFloat64Min = MinPowerEpsFloat64()

#CompareFloat32 = [1.0, (1.0 + ResultFloat32[0] / 2), (1.0 + ResultFloat32[0]),
                #  (1.0 + ResultFloat32[0]/2 + ResultFloat32[0])]
CompareFloat64 = [1.0, 1.0 + ResultFloat64[0] / 2, 1.0 + ResultFloat64[0],
                  1.0 + ResultFloat64[0]/2 + ResultFloat64[0]]

print(1.0, 1.0 + ResultFloat32[0] / 2, 1.0 + ResultFloat32[0],
                  1.0 + ResultFloat32[0]/2 * 3, 1.0 + ResultFloat32[0]/2 * 4, 1.0 + ResultFloat32[0]/2 * 5)


print('              Epsilon             ')
print('FLOAT:  Epsilon = ', ResultFloat32[0])
print('DOUBLE: Epsilon = ', ResultFloat64[0])
print('              Mantissa              ')
print('FLOAT:  Mantissa = ', ResultFloat32[1])
print('DOUBLE: Mantissa = ', ResultFloat64[1])
print('        MaxPower and MinPower        ')
print('FLOAT:  Max = ', ResultFloat32Max, '   Min = ', ResultFloat32Min)
print('DOUBLE: Max = ', ResultFloat64Max, '  Min = ', ResultFloat64Min)
print('               Compare               ')
print('FLOAT: ')
#print(CompareFloat32)
print('DOUBLE: ')
print(CompareFloat64)
