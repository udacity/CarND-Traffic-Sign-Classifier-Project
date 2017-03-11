import cv2
import numpy as np


def collage(src, rows, columns):
    count, height, width, channels = src.shape

    dst = np.full((rows * height, columns * width, channels), 255, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            dst[i * height: (i + 1) * height, j * width: (j + 1) * width] = src[i * columns + j]

    return dst


def gray_normalized(src):
    width, height, channels = src.shape
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(dst)
    dst = np.reshape(dst, (width, height, 1))
    return dst
