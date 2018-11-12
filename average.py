import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

f = open("data.txt", "r")

cvDetector = 0
dnnDetector = 0
lineCount = 0

for line in f.readlines():
    lineCount += 1
    # If a detection is missing skip
    # if line.split()[1] == None or line.split()[2] == None: continue
    try:
        cvDetector +=  float(line.split()[1])
        dnnDetector += float(line.split()[2])
    except:
        continue

cvDetector = cvDetector / lineCount
dnnDetector = dnnDetector / lineCount

objects = ("cvDetector", "dnnDetector")

y_pos = np.arange(len(objects))
y = [cvDetector, dnnDetector]

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Average Distance from Actual")
plt.title("Object Detection Method vs. Average Distance from Actual")

plt.show()