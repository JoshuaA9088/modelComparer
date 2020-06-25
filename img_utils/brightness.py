import os

import cv2


def increase_brightness(img, value=30):
    """
    Algorithmically increase the brightness of cv2 images.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_files_by_extension(path, extension):
    xml_list = []
    for filename in os.listdir(path):
        if filename.endswith(extension):
            xml_list.append(os.path.join(path, filename))

    return xml_list


files = get_files_by_extension("imagesBoard/", ".jpg")

for i in files:
    img = cv2.imread(i)
    img = increase_brightness(img, 50)
    cv2.imwrite("new/" + i[12:], img)
    print("new_path: {}".format("new/" + i[12:]))
