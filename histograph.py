import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

cvDetector = []
dnnDetector = []
points = 0

f = open("data.txt", "r")

for i in f.readlines():
    cvDetector.append(i.split()[1])
    dnnDetector.append(i.split()[2])
    points += 1

mu, sigma = 100, 15
# x = mu + sigma*np.random.randn(10000)
x = dnnDetector
# print(x)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([0, 21, 0, .2])
plt.grid(True)

plt.show()