import argparse
import math
import os
from xml.dom.minidom import parse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="Input directory of xml files")

# ap.add_argument("-o", "--output_dir", required=True,
# 				help="Output path of txt file")

args = vars(ap.parse_args())

# Return list of all xml files in path
def get_xml_files(path):
    xml_list = []
    for filename in os.listdir(path):
        if filename.endswith(".xml"):
            xml_list.append(os.path.join(path, filename))
    return xml_list


dom = parse("images.xml")

# Function to parse xml dom object into strings
def parser(element):
    # Grab xml node object
    val = " ".join(t.nodeValue for t in element.childNodes if t.nodeType == t.TEXT_NODE)
    return val


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def writeToTxt(nameDict, filename):
    f = open(filename, "w+")
    # For each class convert coordinates to centroid
    # Then convert to string and write to txt
    for i in nameDict:
        xmin = int(nameDict[i][0])
        xmax = int(nameDict[i][1])
        ymin = int(nameDict[i][2])
        ymax = int(nameDict[i][3])

        # Sample Output:
        # category
        # xmin ymax xmax ymin
        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        x, y, w, h = convert((width, height), b)

        # final = str(i) + "\n" + str(xmin) + " " + str(ymax) + " " + str(xmax) + " " + str(ymin) + "\n"

        final = (
            str(i) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
        )

        f.write(final)

    f.close()


files = get_xml_files(args["input_dir"])
for i in files:
    print(i)

    dom = parse(i)
    # Grab name, width, height, and coordinates from xml
    name = dom.getElementsByTagName("name")
    width = dom.getElementsByTagName("width")
    height = dom.getElementsByTagName("height")

    xmin = dom.getElementsByTagName("xmin")
    xmax = dom.getElementsByTagName("xmax")
    ymin = dom.getElementsByTagName("ymin")
    ymax = dom.getElementsByTagName("ymax")

    # Global floating width and height
    # Only to be written once
    width = int(parser(width[0]))
    height = int(parser(height[0]))

    nameDict = {}

    for j in range(len(name)):
        realName = parser(name[j])
        nameDict[realName] = [
            parser(xmin[j]),
            parser(xmax[j]),
            parser(ymin[j]),
            parser(ymax[j]),
        ]

    filename = i[:-3] + "txt"
    writeToTxt(nameDict, filename)
