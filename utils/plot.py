# coding=utf-8
from math import ceil
from numpy import arange
import matplotlib.pyplot as plt

def plot_train_images(X_train, y_train, n_classes):
    plt.figure(figsize=(20, 15))
    for i in range(0, n_classes):
        plt.subplot(7, 7, i+1)
        x_selected = X_train[y_train == i]
        plt.imshow(x_selected[0, :, :, :])
        plt.title(i)
        plt.axis('off')
    plt.show()

def plot_test_images(test_img, signs_class = None, signs_classes = None, per_line = 4):
    images_count = len(test_img)
    lines = ceil(images_count/per_line)
    plt.figure(figsize=(12, 8))
    for i in range(images_count):
        plt.subplot(lines, per_line, i+1)
        plt.imshow(test_img[i])
        plt.title(signs_class[signs_classes[i]] if signs_class and signs_classes else i + 1)
        plt.axis('off')
    plt.show()

def plot_probabilities(test_img, top5, signs_class):
    plt.figure(figsize=(16, 21))
    for i in range(8):
        plt.subplot(8, 2, 2*i+1)
        plt.imshow(test_img[i])
        plt.title(i)
        plt.axis('off')
        plt.subplot(8, 2, 2*i+2)
        plt.barh(arange(1, 6, 1), top5.values[i, :])
        labs = [signs_class[j] for j in top5.indices[i]]
        plt.yticks(arange(1, 6, 1), labs)
    plt.show()

def plot_learing_progress(f_acc):
    with open(f_acc, 'rt') as accfile:
        accreader = csv.reader(accfile, delimiter=',')
        acc = [row[1] for row in accreader][1:]
        acc.insert(0, 0)
        plt.plot(list(range(0, len(acc))), acc, label='Accuracy on training set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
