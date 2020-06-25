import argparse
import math
import os
from xml.dom.minidom import parse

import cv2

import customModel
import cvDetector

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--input_dir", required=True, help="Input dir of XML files and images"
)

args = vars(ap.parse_args())


def get_files_by_extension(path, extension):
    xml_list = []
    for filename in os.listdir(path):
        if filename.endswith(extension):
            xml_list.append(os.path.join(path, filename))
    return xml_list


def parser(element):
    # Grab xml node object
    val = " ".join(t.nodeValue for t in element.childNodes if t.nodeType == t.TEXT_NODE)
    return val


xmls = get_files_by_extension(args["input_dir"], ".xml")
centroid_radi = 7
# jpgs = get_files_by_extension(args["input_dir"], ".jpg")

for i in xmls:
    dom = parse(i)

    file_list = dom.getElementsByTagName("filename")
    cat_list = dom.getElementsByTagName("name")

    width_list = dom.getElementsByTagName("width")
    height_list = dom.getElementsByTagName("height")

    xmin_list = dom.getElementsByTagName("xmin")
    xmax_list = dom.getElementsByTagName("xmax")
    ymin_list = dom.getElementsByTagName("ymin")
    ymax_list = dom.getElementsByTagName("ymax")

    for j in range(len(file_list)):
        # Parse all the necessary coords and other data
        cat = parser(cat_list[j])
        filename = parser(file_list[j])

        width = int(parser(width_list[j]))
        height = int(parser(height_list[j]))

        xmin = int(parser(xmin_list[j]))
        xmax = int(parser(xmax_list[j]))
        ymin = int(parser(ymin_list[j]))
        ymax = int(parser(ymax_list[j]))

        xCenter = int((xmin + xmax) / 2)
        yCenter = int((ymin + ymax) / 2)

        # Read the file based on the xml
        img = cv2.imread(args["input_dir"] + filename)

        # Manual Data
        cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
        cv2.circle(img, (xCenter, yCenter), centroid_radi, (0, 255, 0), -1)

        # CV Data
        contoursChassis, chassisCentroid = cvDetector.show_video(img)
        cv2.drawContours(img, contoursChassis, -1, (0, 0, 255), 2)
        if chassisCentroid != None:
            cv2.circle(img, chassisCentroid, centroid_radi, (0, 0, 255), -1)

        # Custom Model Dat
        xmin, xmax, ymin, ymax, xCenter, yCenter = customModel.process_img(img)
        cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (255, 0, 0), 2)
        cv2.circle(img, (xCenter, yCenter), centroid_radi, (255, 0, 0), -1)

        # Auto resizer if too big
        if width > 640 or height > 480:
            img = cv2.resize(img, (640, 480))
            print("Auto Resized: %s" % filename)

        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", img)

        c = cv2.waitKey(0)
        if "q" == chr(c & 255):
            exit(0)
