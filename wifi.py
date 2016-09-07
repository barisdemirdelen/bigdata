import pandas as pd
import numpy as np
from scipy.constants import c
from scipy.constants import pi
from scipy.optimize import minimize
from scipy.stats import chi2
from matplotlib import pyplot as plt

dataset = pd.read_csv("UvA-wifitracking-exercise-prepped-data.csv")
# print(dataset)

f = 2.4e9
np.random.seed(100)


def get_transmission_power(Pt, r):
    Pr = Pt + 20 * np.log10(c / (4 * pi * f * r))
    return Pr


def get_transmission_power_coords(Pt, router_pos, wifi_pos):
    r = ((router_pos[0] - wifi_pos[0]) ** 2 + (router_pos[1] - wifi_pos[1]) ** 2 + (
        router_pos[2] - wifi_pos[2]) ** 2) ** 0.5
    return get_transmission_power(Pt, r)


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
plt.hist(normalized_residual, normed=True)
# plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

Pt = np.zeros(1000)
r = 404 ** 0.5
Pr = get_transmission_power(Pt, r) + np.random.normal(0, 2, 1000)

normalized_residual = get_transmission_power(Pt, r) - Pr
plt.clf()
plt.hist(normalized_residual, normed=True)
# plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

Pt = np.zeros(1000)
r = (25 * 25 + 2 * 2) ** 0.5
Pr = Pr = Pt + 25 * np.log10(c / (4 * pi * f * r)) + np.random.normal(0, 1, 1000)

normalized_residual = get_transmission_power(Pt, r) - Pr
plt.clf()
plt.hist(normalized_residual, normed=True)
# plt.show()

mean = np.mean(normalized_residual)
std = np.std(normalized_residual)

print("Mean is %.4f and std is %.4f" % (mean, std))

print("done.")

'''
    Here starts 0.4 Toy Monte Carlo
'''

# k
routers = [(0, 0, 3), (0, 20, 3), (20, 0, 3), (20, 20, 3)]
device_position = (5, 5, 1)
Pt = 0

chi_squareds = []
for x in range(-5, 15):
    estimated_position = [x, 5, 1]
    total = 0
    for router in routers:
        Si = get_transmission_power_coords(Pt, router, device_position)
        Pr = get_transmission_power_coords(Pt, router, estimated_position)
        total += (Si - Pr) ** 2
    chi_squareds.append(total)

plt.clf()
plt.plot(np.arange(-5, 15), chi_squareds)
# plt.show()


# l

plt.clf()
for i in range(5):
    chi_squareds = []
    for x in range(-5, 15):
        estimated_position = [x, 5, 1]
        total = 0
        for router in routers:
            Si = get_transmission_power_coords(Pt, router, device_position) + np.random.normal(0, 1)
            Pr = get_transmission_power_coords(Pt, router, estimated_position)
            total += (Si - Pr) ** 2
        chi_squareds.append(total)

    plt.plot(np.arange(-5, 15), chi_squareds)


# plt.show()


# m

def get_chi_squared(pos_est):
    estimated_position = [pos_est[0], pos_est[1], 1]
    total = 0
    for i, router in enumerate(routers):
        Si = S[i]
        Pr = get_transmission_power_coords(Pt, router, estimated_position)
        total += (Si - Pr) ** 2
    return total


S = []
for router in routers:
    Si = get_transmission_power_coords(Pt, router, device_position) + np.random.normal(0, 1)
    S.append(Si)
x0 = np.array([10.0, 10.0])
# check_grad_result = check_grad(get_chi_squared, get_chi_squared_grad, x0)
# print(check_grad_result)
result = minimize(get_chi_squared, x0, method="L-BFGS-B", jac=False, options={'maxiter': 1000})
print(result)

# n
positions = []
Xs = []
for i in range(1000):
    S = []
    for router in routers:
        Si = get_transmission_power_coords(Pt, router, device_position) + np.random.normal(0, 1)
        S.append(Si)
    x0 = np.array([10.0, 10.0])
    # check_grad_result = check_grad(get_chi_squared, get_chi_squared_grad, x0)
    # print(check_grad_result)
    result = minimize(get_chi_squared, x0, method="L-BFGS-B", jac=False, options={'maxiter': 1000})
    positions.append(result.x)
    Xs.append(get_chi_squared(result.x))

positions = np.array(positions)
Xs = np.array(Xs)

mean_pos = positions.mean(axis=0)
mean_chi_square = Xs.mean()
print("Mean of (x,y): (%.2f,%.2f), mean of chi squared: %.2f" % (mean_pos[0], mean_pos[1], mean_chi_square))

plt.clf()
plt.scatter(positions[:, 0], positions[:, 1])
# plt.show()
plt.clf()
plt.hist(Xs, normed=True)
# plt.show()

# o
# 2 degress of freedom = yes

# p
plt.clf()
df = 2
plt.plot(chi2.pdf(np.arange(16), df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
plt.hist(Xs, normed=True)
# plt.show()

# q
positions = []
Xs = []
for i in range(1000):
    S = []
    for router in routers:
        Si = get_transmission_power_coords(Pt, router, device_position) + np.random.normal(0, 2)
        S.append(Si)
    x0 = np.array([10.0, 10.0])
    # check_grad_result = check_grad(get_chi_squared, get_chi_squared_grad, x0)
    # print(check_grad_result)
    result = minimize(get_chi_squared, x0, method="L-BFGS-B", jac=False, options={'maxiter': 1000})
    positions.append(result.x)
    Xs.append(get_chi_squared(result.x))

positions = np.array(positions)
Xs = np.array(Xs)

mean_pos = positions.mean(axis=0)
mean_chi_square = Xs.mean()
print("Mean of (x,y): (%.2f,%.2f), mean of chi squared: %.2f" % (mean_pos[0], mean_pos[1], mean_chi_square))

plt.clf()
plt.scatter(positions[:, 0], positions[:, 1])
# plt.show()
plt.clf()
plt.hist(Xs, normed=True)
df = 2
plt.plot(chi2.pdf(np.arange(16), df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
plt.hist(Xs, normed=True)
plt.show()

# They didn't fit, our average is much higher

# r
#  We underestimated the uncertainty by a factor of 2

