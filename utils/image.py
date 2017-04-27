# coding=utf-8
import glob
import operator
from csv import reader
from numpy import array, float32, random, uint8, zeros
from cv2 import \
    COLOR_BGR2GRAY, \
    IMREAD_COLOR, \
    INTER_CUBIC, \
    GaussianBlur, \
    add, \
    addWeighted, \
    cvtColor, \
    equalizeHist, \
    getRotationMatrix2D, \
    imread, \
    merge, \
    multiply, \
    resize, \
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
    ang = random.choice((operator.add, operator.sub))(45 * random.rand(), 25 * random.rand())
    Mat = getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return warpAffine(img_in, Mat, img_in.shape[:2])

def zoom_in(img_in, mar=0, img_size=32):
    c_x,c_y, sh = int(img_in.shape[0]/2), int(img_in.shape[1]/2), int(img_size/2-mar)
    img = img_in[(c_x-sh):(c_x+sh),(c_y-sh):(c_y+sh)]
    return resize(img, (img_size, img_size), interpolation = INTER_CUBIC)

def sharpen(img_in):
    gb = GaussianBlur(img_in, (3, 3), 3.0)
    return addWeighted(img_in, 2, gb, -1, 0)

def lin(img_in, s=1.0, m=0.0):
    img = multiply(img_in, array([s]))
    return add(img, array([m]))

def contr(img_in, s=1.0):
    return lin(img_in, s, 127.0*(1.0-s))

def transform(img_in):
    img = zoom_in(img_in, 3)
    img = sharpen(img)
    return eq_Hist(img)

def augment(img_in):
    img = rotate(img_in)
    return transform(img)

def gray(img_in):
    return cvtColor(img_in, COLOR_BGR2GRAY)

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

def load_test_images(path, f_name, need_gray = False):
    test_img = uint8(zeros((8, 32, 32, 3)))
    for i, imp in enumerate([p for p in glob.glob(path)]):
        img = imread(imp, IMREAD_COLOR)
        b, g, r = split(img)
        img = merge([r, g, b])
        if need_gray:
            img = gray(img)
        test_img[i] = transform(img)

    with open(f_name, 'rt') as dfile:
        r = reader(dfile, delimiter=',')
        test_sign_class = [ row[0] for row in r ][1:]
        test_sign_class = [ int(i) for i in test_sign_class ]

    return test_img, test_sign_class
