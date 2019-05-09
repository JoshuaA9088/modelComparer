import argparse
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input path of data file")

args = vars(ap.parse_args())

try:
    f = open(args["input_dir"], "r")
except FileNotFoundError or FileExistsError:
    print("Input path not valid")
    exit(0)

lineCount = 0

cvChassis = []
dnnChassis = []
cvBoard = []
dnnBoard = []

for line in f.readlines():
    lineCount += 1
    # If a detection is missing skip
    # if line.split()[1] == None or line.split()[2] == None: continue
    try:
        cvChassis.append(float(line.split()[1]))
        cvBoard.append(float(line.split()[2]))
        dnnChassis.append(float(line.split()[3]))
        dnnBoard.append(float(line.split()[4]))
    except:
        continue

cvChassis = sum(cvChassis) / lineCount
dnnChassis = sum(dnnChassis) / lineCount
cvBoard = sum(cvBoard) / lineCount
dnnBoard = sum(dnnBoard) / lineCount
# dnnDetector = dnnDetector / lineCount

objects = ("cvChassis", "dnnChassis", "cvBoard", "dnnBoard")

y_pos = np.arange(len(objects))
y = [cvChassis, dnnChassis, cvBoard, dnnBoard]

# print("Average CV: %d" % cvDetector)
# print("Average DNN: %d" % dnnDetector)

# print("Max CV: %d" % max(cvDetector_list))
# print("Max DNN: %d" % max(dnnDetector_list))

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Average Distance from Actual (# of Pixels)\n (Lower is Better)")
plt.title("Object Detection Method vs. Average Distance from Actual")

plt.show()
