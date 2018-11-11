from myro import *
import cv2
import numpy as np

URL = "http://10.0.0.101:8000/stream.mjpg"
cap = cv2.VideoCapture(URL)
init("com4")

setAngle(0)

for i in range(0,360,2):
    turnTo(i, "deg")
    ret, frame = cap.read()
    cv2.imwrite("test/frameW{}.jpg".format(i), frame)
    wait(3)
