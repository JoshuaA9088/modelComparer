import math
import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input dir of Txt files and images")

args = vars(ap.parse_args())

def get_files_by_extension(path, extension):
    file_list = []
    for filename in os.listdir(path):
        if filename.endswith(extension):
            file_list.append(os.path.join(path, filename))
    return file_list

txts = get_files_by_extension(args["input_dir"], ".txt")
jpgs = get_files_by_extension(args["input_dir"], ".jpg")

list_word = []
file_coord_dict = {}
for i in txts:
    f = open(i, "r")
    for l in f.readlines():
        list_word.append(l.split(" "))
    file_coord_dict[i] = list_word
    print(file_coord_dict)

for i in range(len(file_coord_dict)):
    print(file_coord_dict[i])

# print(*file_coord_dict, sep='\n')

"""
[[chassis], [qr]]

a





"""