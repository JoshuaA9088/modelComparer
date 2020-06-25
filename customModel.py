import os
import platform
import sys
import threading
import time
from collections import defaultdict
from io import StringIO

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
MODEL_NAME = "graph_144k/"
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


# config = tf.ConfigProto()
# threads = 4
# print("Intel CPU Detected...")
# print("Attempting to configure Intel MKL DNN...")
# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = threads # SHOULD ALWAYS BE SAME AS OMP_NUM_THREADS
# config.inter_op_parallelism_threads = threads
# os.environ["OMP_NUM_THREADS"] = str(threads)
# os.environ["KMP_BLOCKTIME"] = str(threads)
# os.environ["KMP_SETTINGS"] = "0"
# os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# print("Successfully loaded Intel MKL Settings")


def process_img(image_np):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

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

            # Chassis Centroid
            if scores[0] > 0.5:
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

                # To be added back soon

                # QR Code Centroid
                # if scores[1] > .5:
                #     box = tuple(boxes[1].tolist())
                #     yMin = box[0] * height
                #     xMin = box[1] * width
                #     yMax = box[2] * height
                #     xMax = box[3] * width

                #     xCenter = (xMax + xMin) / 2
                #     yCenter = (yMax + yMin) / 2

                #     xCenterQr = int(xCenter)
                #     yCenterQr = int(yCenter)


if __name__ == "__main__":
    path = "images/frameColor2.jpg"
    im = cv2.imread(path)

    print(process_img(im))
