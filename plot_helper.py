import cv2
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib figure size to have large enough image plotting
FIGURE_SIZE = (12, 6)

def plot_rgb(img):
    """ A helper for plotting a BGR image with matplotlib """
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(img)
    
def plot_bgr(img):
    """ A helper for plotting a BGR image with matplotlib """
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
def plot_gray(gray):
    """ A helper for plotting a grayscale image with matplotlib """
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(gray, cmap='gray')
    
def to_image(img):
    """ Expects a 2D numpy array of any type, representing an image """
    """ Returns an np.uint8 array with range 0..255 """
    fl = img.astype(np.float32)
    minimum = fl.min()
    maximum = fl.max()
    range_ = (maximum - minimum)
    normalized = (((fl - minimum) / range_ ) * 255.).astype(np.uint8)
    return normalized
