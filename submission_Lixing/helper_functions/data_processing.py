import math
import random
import pickle
import matplotlib.pyplot as plt
import csv
import numpy as np
import skimage.transform as st
from PIL import Image


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_training(rootpath, split_rate=0.714):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark,
    then reshape them into 32*32 inputs,
    then split them into training set and validation set.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    data_dict = {}
    # loop over all 42 classes
    print("Reading original data, please wait......")
    for c in range(1):
        print("Reading directory number", c, "......")
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            img_new = plt.imread(prefix + row[0])

            images.append(img_new)  # the 1th column is the filename
            item_new = {"features": img_new[np.newaxis, :],
                        "labels": int(row[7]),
                        "sizes": (int(row[1]), int(row[2])),
                        "coords": (int(row[3]), int(row[4]), int(row[5]), int(row[6]))}
            item_reshaped = img_dict_reshape(item_new)
            data_dict[format(c, '05d') + '_' + row[0]] = item_reshaped
        gtFile.close()
    print("Successfully load all images!")
    dict_training, dict_test = split_dict(data_dict)
    return images, dict_training, dict_test


def readTrafficSigns_test(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    data_dict = {}
    # loop over all 42 classes
    print("Reading original test data, please wait......")
    gtFile = open(rootpath + '/GT-final_test.test.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        img_new = plt.imread(rootpath + '/' + row[0])

        images.append(img_new)  # the 1th column is the filename
        item_new = {"features": img_new[np.newaxis, :],
                    # "labels": int(row[7]),
                    "sizes": (int(row[1]), int(row[2])),
                    "coords": (int(row[3]), int(row[4]), int(row[5]), int(row[6]))}
        item_reshaped = img_dict_reshape(item_new)
        data_dict[row[0]] = item_reshaped
    gtFile.close()
    print("Successfully loaded all test images!")
    return images, data_dict


def split_dict(original_dict, split_rate=0.714):
    """
    Split a dictionary into 2 sets, according to the split rate.
    :param dict: a dictionary
    :param split_rate: a number between 0 and 1
    :return dict_1: with split_rate
    :return dict_2: with (1-split_rate)
    """
    len_dict = len(original_dict)
    len_dict_1 = round(len_dict * split_rate)
    key_list = list(original_dict.keys())
    set_index = set(range(len_dict))
    set_index_1 = set(random.sample(range(len_dict), len_dict_1))
    set_index_2 = set_index - set_index_1
    dict_1 = dict()
    dict_2 = dict()
    for i in set_index_1:
        dict_1[key_list[i]] = original_dict[key_list[i]]
    for j in set_index_2:
        dict_2[key_list[j]] = original_dict[key_list[j]]
    return dict_1, dict_2


def img_dict_reshape(img_dict, size_new=(32, 32)):
    """
    Reshapes an image dictionary to a new size, so that a new image array and bounding box coordinates are returned.
    :param img_dict: a a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    :param size_new: the expected size tuple (width, height)
    :return: a new dictionary with the same structure as img_dict
    """
    width_old = img_dict["sizes"][0]
    height_old = img_dict["sizes"][1]
    width_new, height_new = size_new

    img_reshaped = np.round(st.resize(img_dict["features"][0], size_new) * 255).astype(np.uint8)
    img_dict["features"] = img_reshaped[np.newaxis, :]

    x1_new = math.floor(img_dict["coords"][0] / width_old * width_new)
    y1_new = math.floor(img_dict["coords"][1] / height_old * height_new)
    x2_new = math.ceil(img_dict["coords"][2] / width_old * width_new)
    y2_new = math.ceil(img_dict["coords"][3] / height_old * height_new)
    img_dict["sizes"] = size_new
    img_dict["coords"] = (x1_new, y1_new, x2_new, y2_new)

    return img_dict


def img_dict_rgb2gray(img_dict, show=False):
    """
    Extract a channel from the rgb image, so that a width*height*1 matrix is achieved.
    :param img_dict: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    :param channel: available channels: "GRAY"
    :return: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    """
    img_1 = Image.fromarray(img_dict["features"][0])
    img_new = img_1.convert("L")
    img_new_mat = np.array(img_new)
    img_dict["features"] = img_new_mat[np.newaxis, :]
    if show:
        img_dict_show(img_dict)
    return img_dict


def img_array_rgb2gray(img_arr, show=False):
    """
    Extract a channel from the rgb image, so that a width*height*1 matrix is achieved.
    :param show: whether to show the image
    :param img_arr: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    :return: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    """
    img_1 = Image.fromarray(img_arr)
    img_new = img_1.convert("L")
    img_new_mat = np.array(img_new)
    if show:
        img_new.show()
    return img_new_mat


def img_dict_show(img_dict):
    """
    Show an image from a img dictionary.
    :param img_dict: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    :return: a dictionary with 4 key/value pairs: "features", "labels", "sizes", "coords"
    """
    img_1 = Image.fromarray(np.array(img_dict["features"][0]))
    img_1.show()


def data_pickle(imgs_dict, save_path="./data.txt"):
    """This is used to pickle the generated images dictionary into a readable file for future usage."""
    print("pickling images, please wait...")
    with open(save_path, 'wb') as f:
        pickle.dump(imgs_dict, f, 0)
    print("successfully pickled, saved as", save_path)
    return


if __name__ == "__main__":
    root_training = "../datasets/official_data/training/Images"
    _, dict_training, dict_validation = readTrafficSigns_training(root_training)
    # data_pickle(dict_training, "../datasets/official_data/data_training.txt")
    # data_pickle(dict_validation, "../datasets/official_data/data_validation.txt")
    sample_dict = list(dict_validation.values())[0]
    img_dict_rgb2gray(list(dict_validation.values())[0], show=True)
    """
    root_test = "../datasets/official_data/test/Images"
    _, data_dict = readTrafficSigns_test(root_test)
    data_pickle(data_dict, "../datasets/data_test.txt")
    """