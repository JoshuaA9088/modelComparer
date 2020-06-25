### Usage: python findRobot.py IP_ADDRESS COMX
import math
import sys
import threading
import time

import cv2
import numpy as np

# Handles click on Original Picture
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pt
        pt = (x, y)
        return pt


def calibrate(img):
    if type(img) == str:
        img = cv2.imread(img)
    chassisImg = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    cv2.imshow("Calibration Image", chassisImg)
    cv2.waitKey(0)


def show_video(jpg, draw=False):
    redUpper = np.array(
        [100, 145, 150], dtype=np.uint8
    )  # Upper threshold for chassis ID
    redLower = np.array(
        [50, 100, 120], dtype=np.uint8
    )  # Lower threshold for chassis ID

    greenUpper = np.array([0, 200, 100], dtype=np.uint8)  # Upper threshold for board ID
    greenLower = np.array([0, 0, 0], dtype=np.uint8)  # Lower threshold for board ID

    kernel = np.ones((5, 5), np.uint8)

    # YUV and LUV Work really well here, currenty sets everything robot to white
    readColors = jpg

    global origPic, chassisImg, boardImg
    origPic = readColors  # Keeps an original unedited
    chassisImg = cv2.cvtColor(
        readColors, cv2.COLOR_BGR2LUV
    )  # Converts to LUV for chassis detection
    # chassisImg = origPic.copy()
    boardImg = origPic.copy()  # Copies raw RGB imgae to use for board / strip detection

    blurredImgChassis = cv2.GaussianBlur(
        chassisImg, (11, 11), 10
    )  # Blurs image to deal with noise
    maskChassis = cv2.inRange(
        blurredImgChassis, redLower, redUpper
    )  # Creates blob image based on threshold; redLower and redUpper
    maskChassis = cv2.erode(
        maskChassis, kernel, iterations=2
    )  # Erodes to get rid of random specks
    maskChassis = cv2.dilate(
        maskChassis, kernel, iterations=2
    )  # Dialates to get rid of random specks

    blurredImgBoard = cv2.GaussianBlur(
        boardImg, (11, 11), 10
    )  # Blurs image to deal with noise
    maskBoard = cv2.inRange(
        blurredImgBoard, greenLower, greenUpper
    )  # Creates blob image based on threshold; greenLower and greenUpper
    maskBoard = cv2.erode(
        maskBoard, kernel, iterations=2
    )  # Erodes to get rid of random specks
    maskBoard = cv2.dilate(
        maskBoard, kernel, iterations=2
    )  # Dialates to get rid of random specks

    edgeChassis = cv2.Canny(
        maskChassis, 75, 200
    )  # Runs cv2.canny to give us better contours
    edgeBoard = cv2.Canny(
        maskBoard, 75, 200
    )  # Runs cv2.canny to give us better contours

    im2Chassis, contoursChassis, hierarchyChassis = cv2.findContours(
        edgeChassis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # Find countour for masked chassisimage
    im2Board, contoursBoard, hierarchyBoard = cv2.findContours(
        edgeBoard, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # Find countour for masked borad image

    # cv2.drawContours(chassisImg, contoursChassis, -1, (0,255,0), 2) #Draw countours on alternate color space chassis image
    # cv2.drawContours(boardImg, contoursBoard, -1, (0,255,0), 2) #Draw countours on alternate color space board image

    if len(contoursChassis) != 0:
        c = max(contoursChassis, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(chassisImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if draw != False:
        cv2.imshow("chassis", chassisImg)
        cv2.imshow("Original", origPic)

        cv2.waitKey(0)

    try:
        chassisCentroid = calcCentroids(contoursChassis)
        return contoursChassis, chassisCentroid
    except:
        return contoursChassis, None


### Centroid Calculations ###
# All centroid calculations use the picked contours #
def calcCentroids(contour_list_chassis):
    for contours in contour_list_chassis:
        mChassis = cv2.moments(contours)
        cxC = int(
            mChassis["m10"] / mChassis["m00"]
        )  # Centroid Calculation for x chassis
        cyC = int(
            mChassis["m01"] / mChassis["m00"]
        )  # Centroid Calculation for y chassis
        centroidChassis = (cxC, cyC)

    return centroidChassis


if __name__ == "__main__":

    # path = "images/frameColorW56.jpg"
    # path = "images/frameColor238.jpg"
    path = "images/frame192.jpg"

    im = cv2.imread(path)
    contoursChassis, chassisCentroid = show_video(im)
    cv2.circle(im, chassisCentroid, 7, (0, 0, 255), -1)
    cv2.drawContours(im, contoursChassis, -1, (0, 255, 0), 2)

    cv2.imshow("Img", im)
    cv2.waitKey(0)
