import cv2
from collections import OrderedDict
import argparse
import os
import numpy as np

# Import old method of detection
import cvDetector

# DNN Specific Imports
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input dir of XML files and images")

ap.add_argument("-t", "--input_txt", required=True,
                help="Input dir TXT txt file of coordinates")
args = vars(ap.parse_args())

def calculateDistance(x1, y1, x2, y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

# DNN Necessary Variables
CWD_PATH = os.getcwd()
MODEL_NAME = 'scribbler_graph_board_v3/'
PATH_TO_CKPT = '{}frozen_inference_graph.pb'.format(MODEL_NAME)
PATH_TO_LABELS = 'object-detection.pbtxt'
NUM_CLASSES = 2

# Load Detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)

def customModel(image_np):
    print('ran')
    thresh = .4
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    height, width, channels = image_np.shape

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # Chassis Centroid
    if scores[0] > thresh:
        box = tuple(boxes[0].tolist())
        yMinChassis = int(box[0] * height)
        xMinChassis = int(box[1] * width)
        yMaxChassis = int(box[2] * height)
        xMaxChassis = int(box[3] * width)

        xCenterChassis = int((xMaxChassis + xMinChassis) / 2)
        yCenterChassis = int((yMaxChassis + yMinChassis) / 2)

        chassisDict = {"xMinChassis": xMinChassis, "yMinChassis" : yMinChassis, "xMaxChassis": xMaxChassis, "yMaxChassis" : yMaxChassis, "xCenterChassis" : xCenterChassis, "yCenterChassis" : yCenterChassis}
        # xCenterChassis = int(xCenterChassis)
        # yCenterChassis = int(yCenterChassis)

        # return xMin, xMax, yMin, yMax, xCenterChassis, yCenterChassis
    if scores[1] > thresh:
            box = tuple(boxes[0].tolist())
            yMinBoard = int(box[0] * height)
            xMinBoard = int(box[1] * width)
            yMaxBoard = int(box[2] * height)
            xMaxBoard = int(box[3] * width)

            xCenterBoard = int((xMaxBoard + xMinBoard) / 2)
            yCenterBoard = int((yMaxBoard + yMinBoard) / 2)

            boardDict = {"xMinBoard": xMinBoard, "yMinBoard" : yMinBoard, "xMaxBoard": xMaxBoard, "yMaxBoard" : yMaxBoard, "xCenterBoard" : xCenterBoard, "yCenterBoard" : yCenterBoard}
            # xCenterChassis = int(xCenter)
            # yCenterChassis = int(yCenter)

    if scores[0] > thresh :# and scores[1] > thresh:
        ret = {**chassisDict, **boardDict}
        print(ret)
        return ret

f = open(args['input_txt'], "r")

filenames = []
centroid_radi = 7

with detection_graph.as_default():
       with tf.Session(graph=detection_graph) as sess:
        for i in f.readlines():
            l = i.split()
            filenames.append(l[0])
            for j in range(1, len(l), 5):
                if l[j] == 'chassis':
                    xminChassis = int(l[j+1])
                    yminChassis = int(l[j+2])
                    xmaxChassis = int(l[j+3])
                    ymaxChassis = int(l[j+4])
                    xCenterChassis = int((xminChassis + xmaxChassis) / 2)
                    yCenterChassis = int((yminChassis + ymaxChassis) / 2)
                if l[j] == 'board':
                    xminBoard = int(l[j+1])
                    yminBoard = int(l[j+2])
                    xmaxBoard = int(l[j+3])
                    ymaxBoard = int(l[j+4])
                    xCenterBoard = int((xminBoard + xmaxBoard) / 2)
                    yCenterBoard = int((yminBoard + ymaxBoard) / 2)

            img = cv2.imread(args['input_dir'] + l[0])
            origPic = cv2.imread(args['input_dir'] + l[0])

            height, width, c = img.shape
            if width > 640 or height > 480:
                continue
            
            cv2.rectangle(img, (xminChassis, ymaxChassis), (xmaxChassis, yminChassis), (0, 255, 0), 2)
            cv2.rectangle(img, (xminBoard, ymaxBoard), (xmaxBoard, yminBoard), (0, 255, 0), 2)
            cv2.circle(img, (xCenterChassis, yCenterChassis), centroid_radi, (0, 255,0), -1)
            cv2.circle(img, (xCenterBoard, yCenterBoard), centroid_radi, (0, 255, 0), -1)

            dicts = customModel(img.copy())
            if dicts != None:
                cv2.rectangle(img, (dicts['xMinChassis'], dicts['yMinChassis']), (dicts['xMinChassis'], dicts['xMinChassis']), (255,0,0), -1)
                cv2.imshow('img', img)
            print(dicts)

            c = cv2.waitKey(0)
            if 'q' == chr(c & 255):
                exit(0)

f.close()
