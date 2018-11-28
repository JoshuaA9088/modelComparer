import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import statistics 

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

cvStd = statistics.stdev(cvDetector)
cvAverage = statistics.mean(cvDetector)
print(cvStd)

# An "interface" to matplotlib.axes.Axes.hist() method

# Iterate through both types
for i in range(len(cats)):
        # Generate x Histogram data
        n, bins, patches = plt.hist(x=cats[i], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
        # Y grid creation
        plt.grid(axis='y', alpha=0.75)
        # Label the axis/title
        plt.xlabel('Distance from actual (# Pixels)')
        plt.ylabel('Frequency')
        plt.title('{} Histogram'.format(catNames[i]))
        # Appropriately scale the y axis
        maxfreq = n.max()
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()

f.close()

