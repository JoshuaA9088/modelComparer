"""
Coordinate generator...
Outputs txt with:
FILENAME CATEGORY XMIN YMIN XMAX YMAX CATEGORY XMIN ... -->
"""

import xmltodict
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input dir of XML files and images")

ap.add_argument("-o", "--output_dir", required=True,
                help="Output path of txt")

args = vars(ap.parse_args())

def get_files_by_extension(path, extension):
    xml_list = []
    for filename in os.listdir(path):
        if filename.endswith(extension):
            xml_list.append(os.path.join(path, filename))
    return xml_list


# with open('test/test.xml') as fd:
#     doc = xmltodict.parse(fd.read())
path = args['input_dir']
xmls = get_files_by_extension(path, ".xml")

f = open(args['output_dir'], 'w+')
for j in xmls:
    try:
        doc = xmltodict.parse(open(j).read())
        names = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        filename = doc['annotation']['filename']
        for i in doc['annotation']['object']:
            names.append(dict(i)['name'])
            xmin.append(dict(i)['bndbox']['xmin'])
            ymin.append(dict(i)['bndbox']['ymin'])
            xmax.append(dict(i)['bndbox']['xmax'])
            ymax.append(dict(i)['bndbox']['ymax'])
            
        for n in range(len(names)):
            if n > 0:
                final = " ".join([names[n], xmin[n], ymin[n], xmax[n], ymax[n]]) + " "
                if n == max(range(len(names))):
                    final = final + "\n"
            else:
                final = " ".join([str(filename), names[n], xmin[n], ymin[n], xmax[n], ymax[n]]) + " "
            f.write(final)# + "\n")
    except:
        continue
f.close()