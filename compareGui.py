"""
Compares tradition CV2 color object detector and
DNN object detector

Green bounding box / centroid = User defined (Correct)
Blue bounding box / centroid = DNN Object Detector (Prediction)
Red contours and bounding box / centroid = CV2 Color Object Detector (Prediction)
"""

from xml.dom.minidom import parse
import math
import argparse
import os
import cv2
import numpy as np
from termcolor import colored

# Import old method of detection
import cvDetector

# DNN Specific Imports
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Parse input image folder
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input dir of XML files and images")

ap.add_argument("-m", "--model_path", required=False,
                help="Path to DNN model, defaults to 144k")

args = vars(ap.parse_args())

# returns list of all files that end with ".x"
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

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

# DNN Necessary Variables
CWD_PATH = os.getcwd()
MODEL_NAME = 'scribbler_graph_board_v2/'
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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)

def customModel(image_np):
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
    if scores[0] > .5:
        box = tuple(boxes[0].tolist())
        yMin = int(box[0] * height)
        xMin = int(box[1] * width)
        yMax = int(box[2] * height)
        xMax = int(box[3] * width)

        xCenter = int((xMax + xMin) / 2)
        yCenter = int((yMax + yMin) / 2)

        xCenterChassis = int(xCenter)
        yCenterChassis = int(yCenter)

        return xMin, xMax, yMin, yMax, xCenterChassis, yCenterChassis

# Grab all xmls from input dir
xmls = get_files_by_extension(args["input_dir"], ".xml")
# How big should centroid radiuses be
centroid_radi = 7 

# Initialize DNN graph once to
# lower latency
with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:         
            for i in xmls:
                dom = parse(i)
                
                file_list = dom.getElementsByTagName('filename')
                cat_list = dom.getElementsByTagName('name')
                
                width_list = dom.getElementsByTagName("width")
                height_list = dom.getElementsByTagName("height")

                xmin_list = dom.getElementsByTagName('xmin')
                xmax_list = dom.getElementsByTagName('xmax')
                ymin_list = dom.getElementsByTagName('ymin')
                ymax_list = dom.getElementsByTagName('ymax')
                
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
                    origPic = img.copy()

                    # Manual Data - Correct
                    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0,255,0), 2)
                    cv2.circle(img, (xCenter, yCenter), centroid_radi, (0, 255,0), -1)

                    # CV Data - Prediction
                    contoursChassis, chassisCentroid = cvDetector.show_video(img)
                    c = max(contoursChassis, key=cv2.contourArea)
                    cv2.drawContours(img, c, -1, (0,0,255), 2)
                    x, y, w, h = cv2.boundingRect(c)

                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    chassisCentroid = (int((x+x+w) / 2), int((y+y+h) / 2))
                    cv2.circle(img, chassisCentroid, centroid_radi, (0,0,255), -1)

                    # Custom Model Data - Prediction
                    try:
                        xminDNN, xmaxDNN, yminDNN, ymaxDNN, xCenterDNN, yCenterDNN = customModel(origPic)
                        cv2.rectangle(img, (xminDNN, ymaxDNN), (xmaxDNN, yminDNN), (255,0,0), 2)
                        cv2.circle(img, (xCenterDNN, yCenterDNN), centroid_radi, (255, 0,0), -1)
                    except:
                        pass

                    # Auto resizer if too big
                    if width > 640 or height > 480:
                        img = cv2.resize(img, (640, 480))
                        continue
                        # print("Auto Resized: %s" % filename)

                    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                    cv2.imshow("Original", img)

                    print(colored("Image: {}".format(filename), color="green"))
                    # CV2 Distance from actual
                    try:
                        cvDistance = calculateDistance(chassisCentroid[0], chassisCentroid[1], xCenter, yCenter)
                        print(colored("CV2 Distance: ", color="red"), cvDistance)
                    except:
                        print(colored("CV2 Distance: N/A", color="red"))

                    # DNN Distance from actual
                    try:
                        dnnDistance = calculateDistance(xCenterDNN, yCenterDNN, xCenter, yCenter)
                        print(colored("DNN Distance: ", color="blue"), dnnDistance)
                    except:
                        print(colored("DNN Distance: ", color="blue"))

                    print("\n")
                    # Keypress moves to next image, q exits
                    c = cv2.waitKey(0)
                    if 'q' == chr(c & 255):
                        exit(0)
