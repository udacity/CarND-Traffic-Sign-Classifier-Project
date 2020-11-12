"""
This module handles everything to do with the image files.
"""

from PIL import Image as img
from PIL import ImageDraw as imgd
import matplotlib.pyplot as plt
import skimage.transform as st
import numpy as np
import csv
import os

impath = "../datasets/official_data/training/Images/00028/00002_00025.ppm"


def original_image_show(root, file_name, bounding_box=True, preview=False, save=False, save_path="./"):
    img_1 = img.open(root + file_name)
    if bounding_box:
        img_box = imgd.Draw(img_1)
        dir_name = root.split("/")[-2]
        gtFile = open(root + 'GT-' + dir_name + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        for row in gtReader:
            if row[0] == file_name:
                x1 = int(row[3])
                y1 = int(row[4])
                x2 = int(row[5])
                y2 = int(row[6])
                break
        img_box.line([(x1,y1), (x1,y2), (x2,y2), (x2,y1), (x1,y1)], fill=(0, 255, 0))

    if preview:
        img_1.show()

    if save:
        file_name_main = file_name.split('.')[0]
        img_1.save(save_path + file_name_main + ".jpeg")



def img_preprocess_lenet(img_arr, size_new=(32, 32), grayscale=True):
    """
    Preprocess an image array to a new size, so that a new image array is returned.
    :param img_arr: an image array of size 1 x height x width x 3
    :param size_new: the expected size tuple (width, height)
    :return: a new image array of size 1 x height_new x width_new x 1
    """
    width_new, height_new = size_new

    img_reshaped = np.round(st.resize(img_arr[0], size_new) * 255).astype(np.uint8)
    img_arr_new = img_reshaped[np.newaxis, :]
    if grayscale:
        img_arr_new = img_dict_rgb2gray(img_arr_new)
    return img_arr_new


def img_dict_rgb2gray(img_arr):
    """
    Extract a channel from the rgb image, so that a width*height*1 matrix is achieved.
    :param img_arr: an image array of size 1 x height x width x 1
    :return: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    """
    img_gray = img.fromarray(img_arr[0]).convert("L")
    img_new_mat = np.array(img_gray)
    img_new = img_new_mat[np.newaxis, :, :]#, np.newaxis]
    return img_new


if __name__=="__main__":
    path_dir = "../references/test_image/"
    for filename in sorted(os.listdir(path_dir)):
        name_prefix = filename.split('.')[0]
        print("Processing image:", filename, "......")
        im = plt.imread(path_dir + filename)[np.newaxis, :, :, :]
        im_preprocessed = img_preprocess_lenet(im)
        im_mat = img.fromarray(im_preprocessed[0])
        im_mat.save(path_dir + name_prefix + "_gray.jpg")
    # original_image_show("../datasets/official_data/training/Images/00028/", "00002_00025.ppm", preview=True, save=True)