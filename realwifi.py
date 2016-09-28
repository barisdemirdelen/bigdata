import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from scipy.constants import c
from scipy.constants import pi
from scipy.optimize import minimize
from scipy.stats import chi2

# 6
f = 2.4e9

np.random.seed(100)

routers = {"Lima": (5.82, 5.48, 3.0), "Mike": (11.33, 9.43, 3.0), "Kilo": (12.39, 6.77, 3.0),
           "Oscar": (2.48, 7.36, 3.0), "Alpha": (8.53, 2.16, 3.0), "India": (5.82, 5.48, 3.0),
           "November": (8.34, 4.13, 3.0), "Hotel": (5.43, 4.71, 3.0), "Romeo": (10.99, 5.94, 3.0),
           "Quebec": (6.82, 9.78, 3.0), "Papa": (9.9, 10.39, 3.0)}
device_height = 1.0


def get_transmission_power(Pt, r):
    Pr = Pt + 20 * np.log10(c / (4 * pi * f * r))
    return Pr


def get_transmission_power_coords(Pt, router_pos, wifi_pos):
    r = ((router_pos[0] - wifi_pos[0]) ** 2 + (router_pos[1] - wifi_pos[1]) ** 2 + (
        router_pos[2] - wifi_pos[2]) ** 2) ** 0.5
    return get_transmission_power(Pt, r)


dataset = pd.read_csv("UvA-wifitracking-exercise-prepped-data.csv")
plt.clf()
plt.scatter(dataset["measurementTimestamp"], dataset["seqNr"])
# plt.show()

packet_groups = []
for new_packet in dataset.iterrows():
    new_packet = new_packet[1]

    for packets in packet_groups:
        existing_packet = packets[0]
        if new_packet["seqNr"] == existing_packet["seqNr"] and new_packet["typeNr"] == existing_packet["typeNr"] and \
                        new_packet["subTypeNr"] == existing_packet["subTypeNr"] and abs(
                    new_packet["measurementTimestamp"] - existing_packet["measurementTimestamp"]) < 1000:
            packets.append(new_packet)
            break
    else:
        packet_groups.append([new_packet])

print(len(packet_groups))
print("hello")


def get_chi_squared(inputs, S, sigma):
    Pt, x_est, y_est = inputs
    estimated_position = [x_est, y_est, device_height]
    total = 0
    for router_name in S.keys():
        Si = S[router_name]
        Pr = get_transmission_power_coords(Pt, routers[router_name], estimated_position)
        total += (Si - Pr) ** 2 / sigma / sigma
    return total


# for packets in packet_groups:
results = []
chi2s = []
num_routers = []
variances = []
normalized_residuals = []
for i, packets in enumerate(packet_groups):
    S = {}
    sigma = 5.5
    # if len(packets) < 4:
    #     continue
    num_routers.append(len(packets))
    for packet in packets:
        # router_position = routers[packet["droneId"]]
        # Pr = packet["signal"]
        S[packet["droneId"]] = packet["signal"]

    x0 = np.array([-20.0, 5.0, 5.0])
    # check_grad_result = check_grad(get_chi_squared, get_chi_squared_grad, x0)
    # print(check_grad_result)
    result = minimize(get_chi_squared, x0, args=(S, sigma), method="L-BFGS-B", jac=False, options={'maxiter': 100000})
    # print(result)
    current_chi2 = get_chi_squared(result.x, S, sigma)

    # if abs(result.x[1]) > 100 or abs(result.x[2]) > 100 or result.x[0] > 1:
    #     continue

    Pt, x, y = result.x[0], result.x[1], result.x[2]
    x_variance = 0
    y_variance = 0
    for router_name in S:
        router = routers[router_name]
        x_variance += (-20.0 * (x - router[0]) / (
            np.log(10) * ((x - router[0]) ** 2 + (y - router[1]) ** 2 + (device_height - router[2]) ** 2))) ** 2
        y_variance += (-20.0 * (y - router[1]) / (
            np.log(10) * ((x - router[0]) ** 2 + (y - router[1]) ** 2 + (device_height - router[2]) ** 2))) ** 2

    variances.append((x_variance, y_variance))

    normalized_residual = 0
    for router_name in S:
        normalized_residual += S[router_name] - get_transmission_power_coords(Pt, routers[router_name],
                                                                              (x, y, device_height))
    normalized_residual /= len(routers)
    normalized_residuals.append(normalized_residual)

    chi2s.append(current_chi2)
    # if abs(result.x[1]) > 100:
    #     print("hi dude" + str(i))
    results.append(result.x)

chi2s = np.array(chi2s)
results = np.array(results)
num_routers = np.array(num_routers)
variances = np.array(variances)
normalized_residuals = np.array(normalized_residuals)

mean_routers = np.mean(num_routers)
print("Mean routers per packet: %.3f" % mean_routers)

mean_chi2s = np.mean(chi2s)
mean_results = np.mean(results, axis=0)
print("Mean chi2: %.3f, mean Pt: %.3f, mean position: (%.3f,%.3f)" %
      (mean_chi2s, mean_results[0], mean_results[1], mean_results[2]))

router_positions = []
for router_position in routers.values():
    router_positions.append(router_position)
router_positions = np.array(router_positions)

plt.clf()
plt.scatter(results[:, 1], results[:, 2])
plt.scatter(router_positions[:, 0], router_positions[:, 1], color="r")
# plt.show()

plt.clf()
plt.hist(chi2s, normed=True)
df = 4
plt.plot(chi2.pdf(np.arange(16), df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
plt.hist(chi2s, normed=True)
plt.show()

mean_pos = mean_results[1:]
mean_variance = variances.mean(axis=0)
print("Mean of standard deviations: (%.3f, %.3f)" % (mean_variance[0] ** 0.5, mean_variance[1] ** 0.5))

plt.clf()
ax = plt.subplot()
plt.scatter(results[:, 1], results[:, 2])
plt.scatter(mean_pos[0], mean_pos[1], color="g")
plt.scatter(router_positions[:, 0], router_positions[:, 1], color="r")
for i in range(len(results)):
    ell = Ellipse(xy=(results[i, 1], results[i, 2]),
                  width=variances[i, 0] ** 0.5, height=variances[i, 1] ** 0.5,
                  angle=0, color='g')
    ell.set_alpha(0.5)
    ell.set_facecolor('none')
    ax.add_artist(ell)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
plt.show()

plt.clf()
plt.hist(normalized_residuals)
plt.show()
print("Mean of normalized residuals: %.3f, standard deviation: %.3f" % (
    np.mean(normalized_residuals), np.std(normalized_residuals)))
