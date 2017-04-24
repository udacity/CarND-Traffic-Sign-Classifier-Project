# coding=utf-8
import glob
from numpy import array, float32, random, uint8, zeros
from cv2 import \
    IMREAD_COLOR, \
    GaussianBlur, \
    add, \
    addWeighted, \
    equalizeHist, \
    getRotationMatrix2D, \
    imread, \
    merge, \
    multiply, \
    split, \
    warpAffine

def eq_Hist(img_in):
    img = img_in.copy()
    img[:, :, 0] = equalizeHist(img_in[:, :, 0])
    img[:, :, 1] = equalizeHist(img_in[:, :, 1])
    img[:, :, 2] = equalizeHist(img_in[:, :, 2])
    return img

def rotate(img_in):
    c_x, c_y = int(img_in.shape[0]/2), int(img_in.shape[1]/2)
    ang = 30.0 * random.rand() - 15
    Mat = getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return warpAffine(img_in, Mat, img_in.shape[:2])

def sharpen(img_in):
    gb = GaussianBlur(img_in, (5,5), 15.0)
    return addWeighted(img_in, 2, gb, -1, 0)

def lin(img_in, s=1.0, m=0.0):
    img = multiply(img_in, array([s]))
    return add(img, array([m]))

def contr(img_in, s=1.0):
    return lin(img_in, s, 127.0*(1.0-s))

def transform(img_in):
    img = sharpen(img_in)
    img = contr(img, 1.5)
    return eq_Hist(img)

def augment(img_in):
    img = contr(img_in, 1.8 * random.rand() + 0.2)
    img = rotate(img)
    return transform(img)

# Generating new images from existing ones using transformations
def generate_images(X_train_sub, y_train_sub, X_val, y_val, n = 1):
    n_train_sub = len(y_train_sub)
    n_val = len(y_val)
    X_train_aug = []
    y_train_aug = []
    X_val_prep = []

    for i in range(n_train_sub):
        img = X_train_sub[i]
        X_train_aug.append(transform(img))
        y_train_aug.append(y_train_sub[i])
        for j in range(n):
            X_train_aug.append(augment(img))
            y_train_aug.append(y_train_sub[i])

    for i in range(n_val):
        img = X_val[i]
        X_val_prep.append(transform(img))

    return array(X_train_aug), \
        array(y_train_aug, dtype = float32), \
        array(X_val_prep, dtype = float32)

def load_test_images(path):
    test_img = uint8(zeros((8, 32, 32, 3)))
    for i, imp in enumerate([p for p in glob.glob(path)]):
        img = imread(imp, IMREAD_COLOR)
        b, g, r = split(img)
        img = merge([r, g, b])
        test_img[i] = img
    return test_img
