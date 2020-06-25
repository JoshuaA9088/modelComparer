import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cvDetector_dual as cvDetector

# Import old method of detection
# DNN Specific Imports


ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--input_dir", required=True, help="Input dir of XML files and images"
)

ap.add_argument(
    "-t", "--input_txt", required=True, help="Input dir TXT txt file of coordinates"
)

args = vars(ap.parse_args())


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# DNN Necessary Variables
CWD_PATH = os.getcwd()
MODEL_NAME = "scribbler_graph_board_v3/"
PATH_TO_CKPT = "{}frozen_inference_graph.pb".format(MODEL_NAME)
PATH_TO_LABELS = "object-detection.pbtxt"
NUM_CLASSES = 2

# Load Detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)


def customModel(image_np):
    thresh = 0.4
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

    boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
    scores = detection_graph.get_tensor_by_name("detection_scores:0")
    classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded},
    )

    height, width, channels = image_np.shape

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    for i in range(len(scores)):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())
            yMin = int(box[0] * height)
            xMin = int(box[1] * width)
            yMax = int(box[2] * height)
            xMax = int(box[3] * width)

            xCenter = int((xMax + xMin) / 2)
            yCenter = int((yMax + yMin) / 2)

            cv2.rectangle(image_np, (xMin, yMax), (xMax, yMin), (255, 0, 0), 2)
    return image_np


f = open(args["input_txt"], "r")

filenames = []
centroid_radi = 7

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for i in f.readlines():
            l = i.split()
            filenames.append(l[0])
            for j in range(1, len(l), 5):
                if l[j] == "chassis":
                    xminChassis = int(l[j + 1])
                    yminChassis = int(l[j + 2])
                    xmaxChassis = int(l[j + 3])
                    ymaxChassis = int(l[j + 4])
                    xCenterChassis = int((xminChassis + xmaxChassis) / 2)
                    yCenterChassis = int((yminChassis + ymaxChassis) / 2)
                if l[j] == "board":
                    xminBoard = int(l[j + 1])
                    yminBoard = int(l[j + 2])
                    xmaxBoard = int(l[j + 3])
                    ymaxBoard = int(l[j + 4])
                    xCenterBoard = int((xminBoard + xmaxBoard) / 2)
                    yCenterBoard = int((yminBoard + ymaxBoard) / 2)

            img = cv2.imread(args["input_dir"] + l[0])
            origPic = img.copy()

            height, width, c = img.shape
            if width > 640 or height > 480:
                continue

            img = customModel(img)

            cv2.rectangle(
                img,
                (xminChassis, ymaxChassis),
                (xmaxChassis, yminChassis),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                img, (xminBoard, ymaxBoard), (xmaxBoard, yminBoard), (0, 255, 0), 2
            )
            cv2.circle(
                img, (xCenterChassis, yCenterChassis), centroid_radi, (0, 255, 0), -1
            )
            cv2.circle(
                img, (xCenterBoard, yCenterBoard), centroid_radi, (0, 255, 0), -1
            )

            # CV Data - Prediction
            contoursChassis, contoursBoard = cvDetector.show_video(origPic)
            try:
                # Chassis
                c = max(contoursChassis, key=cv2.contourArea)
                cv2.drawContours(img, c, -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                chassisCentroid = (int((x + x + w) / 2), int((y + y + h) / 2))
                cv2.circle(img, chassisCentroid, centroid_radi, (0, 0, 255), -1)
            except:
                pass

            try:
                # Board
                c = max(contoursBoard, key=cv2.contourArea)
                cv2.drawContours(img, c, -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                chassisBoard = (int((x + x + w) / 2), int((y + y + h) / 2))
                cv2.circle(img, chassisBoard, centroid_radi, (0, 0, 255), -1)
            except:
                pass

            cv2.imshow("img", img)
            print(args["input_dir"] + l[0])
            c = cv2.waitKey(0)
            if "q" == chr(c & 255):
                exit(0)

f.close()
