import numpy as np
from matplotlib import colors
import matplotlib
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
f.close()

fig, ax = plt.subplots()
ax.set_title('cvDetector v. DnnDetector')
ax.boxplot(cvDetector, showfliers=False)
ax.boxplot(dnnDetector, positions=[1,2], showfliers=False)

plt.show()
