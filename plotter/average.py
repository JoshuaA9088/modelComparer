import argparse

import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Input path of data file")

args = vars(ap.parse_args())

try:
    f = open(args["input_dir"], "r")
except FileNotFoundError:
    print("Input path not valid")
    exit(-1)

cvDetector_list = []
dnnDetector_list = []

for line in f.readlines():
    try:
        cvDetector_list.append(float(line.split()[1]))
        dnnDetector_list.append(float(line.split()[2]))
    # Skip malformed data.
    except IndexError:
        continue

cvDetector = sum(cvDetector_list)
dnnDetector = sum(dnnDetector_list)

cvDetector = cvDetector / len(cvDetector)
dnnDetector = dnnDetector / len(dnnDetector)

objects = ("cvDetector", "dnnDetector")

y_pos = np.arange(len(objects))
y = [cvDetector, dnnDetector]

print(f"Average CV: {cvDetector}")
print(f"Average DNN: {dnnDetector}")

print(f"Max CV: {max(cvDetector_list)}")
print(f"Max DNN: {dnnDetector_list}")

plt.bar(y_pos, y, align="center", alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Average Distance from Actual (# of Pixels)")
plt.title("Object Detection Method vs. Average Distance from Actual")

plt.show()
