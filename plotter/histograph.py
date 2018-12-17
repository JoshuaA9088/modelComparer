import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import colors
import statistics 

cvChassis = []
cvBoard = []
dnnChassis = []
dnnBoard = []

points = 0
cats = [cvChassis, cvBoard, dnnChassis, dnnBoard]
catNames = ["cvChassis", "cvBoard", "dnnChassis", "dnnBoard"]
f = open("new.txt", "r")

for i in f.readlines():
    if i.split()[1] != "None" and i.split()[2] != "None" and i.split()[3] != "None" and i.split()[4] != "None":
        cvChassis.append(float(i.split()[1]))
        cvBoard.append(float(i.split()[2]))
        dnnChassis.append(float(i.split()[3]))
        dnnBoard.append(float(i.split()[4]))
        points += 1

# print(len(cvDetector))
# print(len(dnnDetector))
# print(max(cvDetector))
# print(max(dnnDetector))

# Iterate through both types
for i in range(len(cats)):
        mean = np.mean(cats[i])
        std = np.std(cats[i])

        fig, ax = plt.subplots()
        fig.tight_layout()

        # Generate x Histogram data
        n, bins, patches = plt.hist(x=cats[i], bins=50, color='#0504aa',
                                alpha=0.7, rwidth=0.5)
        
        fracs = n / n.max()
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Y grid creation
        plt.grid(axis='y', alpha=0.75)

        # Label the axis/title
        plt.xlabel('Distance from actual (# Pixels)')
        plt.ylabel('Frequency')
        plt.title('{} Histogram'.format(catNames[i]))
        
        # Appropriately scale the axis
        maxfreq = n.max()
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 20)
        plt.xlim(0, 20)
        plt.xticks(np.arange(0, 20, .5), rotation=90)
        plt.show()

f.close()

