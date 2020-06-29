import matplotlib.pyplot as plt

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
f.close()

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(cats, patch_artist=True, showfliers=False)

ax.set_xticklabels(catNames)

# change outline color, fill color and linewidth of the boxes
for box in bp["boxes"]:
    # change outline color
    box.set(color="#7570b3", linewidth=2)
    # change fill color
    box.set(facecolor="#1b9e77")

# change color and linewidth of the whiskers
for whisker in bp["whiskers"]:
    whisker.set(color="#7570b3", linewidth=2)

# change color and linewidth of the caps
for cap in bp["caps"]:
    cap.set(color="#7570b3", linewidth=2)

# change color and linewidth of the medians
for median in bp["medians"]:
    median.set(color="#b2df8a", linewidth=2)

# change the style of fliers and their fill
for flier in bp["fliers"]:
    flier.set(marker="o", color="#e7298a", alpha=0.5)

plt.title("CV v. DNN Box Plot")
plt.xlabel("Type of Detector")
plt.ylabel("Distance from Actual (# of Pixels)\n (Lower is Better)")

plt.show()
