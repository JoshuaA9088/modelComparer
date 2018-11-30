import numpy as np
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import statistics

import T_test

cvDetector = []
dnnDetector = []
points = 0
cats = [cvDetector, dnnDetector]
catNames = ["cvDetector", "dnnDetector"]
f = open("data.txt", "r")

for i in f.readlines():
    if i.split()[1] != "None" and i.split()[2] != "None":
        cvDetector.append(float(i.split()[1]))
        dnnDetector.append(float(i.split()[2]))
        points += 1
f.close()

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(cats, patch_artist=True, showfliers=False)

ax.set_xticklabels(catNames)

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set(color='#7570b3', linewidth=2)
    # change fill color
    box.set(facecolor='#1b9e77')

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)

plt.title("CV v. DNN Box Plot")
plt.xlabel("Type of Detector")
plt.ylabel("Distance from Actual (# of Pixels)")

plt.show()
