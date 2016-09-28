
import matplotlib.pyplot as plt
import pandas as pd

# 6

dataset = pd.read_csv("UvA-wifitracking-exercise-prepped-data.csv")
plt.clf()
plt.scatter(dataset["seqNr"] , dataset["measurementTimestamp"])
plt.show()

