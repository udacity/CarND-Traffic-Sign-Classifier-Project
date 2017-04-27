# coding=utf-8
from math import ceil
from csv import reader
from numpy import arange, array, uint32, linspace, all
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def plot_train_images(X_train, y_train, n_classes, is_gray = False):
    cmap = 'gray' if is_gray else None
    plt.figure(figsize=(20, 15))
    for i in range(0, n_classes):
        plt.subplot(7, 7, i+1)
        x_selected = X_train[y_train == i]
        plt.imshow(x_selected[0, :, :, :], cmap=cmap)
        plt.title(i)
        plt.axis('off')
    plt.savefig('./out/train_images.png')
    plt.show()

def plot_distribution(X_train, y_train, n_classes):
    samples_count = []
    for i in range(0, n_classes):
        x_selected = X_train[y_train == i]
        samples_count.append(len(x_selected))
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, n_classes), samples_count)
    plt.title('Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Images')
    plt.savefig('./out/distribution.png')
    plt.show()


def plot_test_images(test_img, signs_class = None, signs_classes = None, per_line = 4, name = None, is_gray = False):
    has_class = True if signs_class is not None and signs_classes is not None else False
    cmap = 'gray' if is_gray else None
    images_count = len(test_img)
    lines = ceil(images_count/per_line)
    plt.figure(figsize=(12, 8))
    for i in range(images_count):
        plt.subplot(lines, per_line, i+1)
        plt.imshow(test_img[i], cmap=cmap)
        plt.title(signs_class[signs_classes[i]] if has_class else i + 1)
        plt.axis('off')
    plt.savefig('./out/test_images' + ('_' + name if name is not None else '') + '.png')
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
    plt.savefig('./out/probabilities.png')

def plot_learing_progress(f_name):
    with open(f_name, 'rt') as dfile:
        r = reader(dfile, delimiter=',')
        acc = [ row[1] for row in r ][1:]
        dfile.seek(0)
        loss = [ row[2] for row in r ][1:]
        acc.insert(0, 0)
        for item in ((acc, 'Accuracy'), (loss, 'Loss')):
            x = array(range(0, len(item[0])))
            y = [ float(i) for i in item[0] ]
            xnew = linspace(x.min(), x.max(), 100)
            plt.plot(xnew, spline(x, y, xnew), label=item[1] + ' on test set')
            plt.xlabel('Epoch')
            plt.ylabel(item[1])
            plt.legend()
            plt.show()
            plt.savefig('./out/learing_progress_' + item[1].lower() + '.png')

if __name__ == '__main__':
    plot_learing_progress('../logs/acc-loss.csv')
