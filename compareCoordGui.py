import cv2
from collections import OrderedDict


f = open("coordinates.txt", "r")
filenames = []

for i in f.readlines():
    l = i.split()
    filenames.append(l[0])
    for j in range(1, len(l), 5):
        if l[j] == 'chassis':
            xminChassis = int(l[j+1])
            yminChassis = int(l[j+2])
            xmaxChassis = int(l[j+3])
            ymaxChassis = int(l[j+4])
        if l[j] == 'qr':
            xminBoard = int(l[j+1])
            yminBoard = int(l[j+2])
            xmaxBoard = int(l[j+3])
            ymaxBoard = int(l[j+4])

    img = cv2.imread('images/' + l[0])
    cv2.rectangle(img, (xminChassis, ymaxChassis), (xmaxChassis, yminChassis), (0, 255, 0), 2)
    cv2.rectangle(img, (xminBoard, ymaxBoard), (xmaxBoard, yminBoard), (0, 255, 0), 2)
    cv2.imshow('img', img)

    c = cv2.waitKey(0)
    if 'q' == chr(c & 255):
        exit(0)


    

f.close()
