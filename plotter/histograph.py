import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


cv = {
    "chassis": [],
    "board": [],
}

dnn = {
    "chassis": [],
    "board": [],
}

cats = [[v for v in type_.values()] for type_ in (cv, dnn)]
catNames = ["cv_chassis", "cv_board", "dnn_chassis", "dnn_board"]

f = open("new.txt", "r")
for i in f.readlines():
    line = i.split()
    if "None" in i.split():
        continue

    cv["chassis"], cv["board"], dnn["chassis"], dnn["board"] = line


# Iterate through both types
# for i in range(len(cats)):
for i, v in enumerate(cats):
    mean = np.mean(v)
    std = np.std(v)

    fig, ax = plt.subplots()
    fig.tight_layout()

    # Generate x Histogram data
    n, bins, patches = plt.hist(x=v, bins=50, color="#0504aa", alpha=0.7, rwidth=0.5)

    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Y grid creation
    plt.grid(axis="y", alpha=0.75)

    # Label the axis/title
    plt.xlabel("Distance from actual (# Pixels)\n (Lower is Better)")
    plt.ylabel("Frequency")
    plt.title(f"{catNames[i]} Histogram")

    # Appropriately scale the axis
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 20)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 0.5), rotation=90)
    plt.show()

f.close()
