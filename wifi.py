import pandas as pd
import numpy as np
from scipy.constants import c
from scipy.constants import pi
from matplotlib import pyplot as plt

dataset = pd.read_csv("UvA-wifitracking-exercise-prepped-data.csv")
# print(dataset)

f = 2.4e9


def get_transmission_power(Pt, r):
    Pr = Pt + 20 * np.log10(c / (4 * pi * f * r))
    return Pr


def get_distance(Pt, Pr):
    r = c / (10 ** ((Pr - Pt) / 20.0) * 4 * pi * f)
    return r


rs = np.linspace(0.4, 30, 100)
plt.plot(rs, get_transmission_power(0, rs))
# plt.show()

print(get_distance(0, -31) - get_distance(0, -30))
print(get_distance(0, -61) - get_distance(0, -60))

Pt = np.zeros(1000)
r = 404 ** 0.5
Pr = get_transmission_power(Pt, r) + np.random.randn(1000)

normalized_residual = get_transmission_power(Pt, r) - Pr
plt.clf()
plt.hist(normalized_residual)
plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

Pt = np.zeros(1000)
r = 404 ** 0.5
Pr = get_transmission_power(Pt, r) + np.random.normal(0, 2, 1000)

normalized_residual = get_transmission_power(Pt, r) - Pr
plt.clf()
plt.hist(normalized_residual)
plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

Pt = np.zeros(1000)
r = (25 * 25 + 2 * 2) ** 0.5
Pr = Pr = Pt + 25 * np.log10(c / (4 * pi * f * r)) + np.random.normal(0, 1, 1000)

normalized_residual = get_transmission_power(Pt, r) - Pr
plt.clf()
plt.hist(normalized_residual)
plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

print("done.")
