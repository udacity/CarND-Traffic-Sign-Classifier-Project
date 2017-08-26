import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2
from imgaug import augmenters as iaa


########################################################################################################################
######                                        Plot utility functions                                              ######
########################################################################################################################

 
def plot_results(metrics, axis, lbs, xlb, ylb,  title, fig_size=(7, 5), epochs_interval=10):
    """
    Nifty utility function to plot results of the execution of our model
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    total_epochs = metrics[0].shape[0]
    x_values = np.linspace(1, total_epochs, num=total_epochs, dtype=np.int32)
    
    for m, l in zip(metrics, lbs):
        ax.plot(line1 = ax.plot(x_values, m[:, axis], linewidth=2, label=l))
    
    ax.set(xlabel=xlb, ylabel=ylb, title=title)
    ax.xaxis.set_ticks(np.linspace(1, total_epochs, num=int(total_epochs/epochs_interval), dtype=np.int32))
    ax.legend(loc='lower right')
    plt.show()
    
def plot_model_results(metrics, axes, lbs, xlb, ylb, titles, fig_title, fig_size=(7, 5), epochs_interval=10):
    """
    Nifty utility function to plot results of the execution of our model
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(axes), figsize=fig_size)
    print("Length of axis: {0}".format(axs.shape))
    
    total_epochs = metrics[0].shape[0]
    x_values = np.linspace(1, total_epochs, num=total_epochs, dtype=np.int32)
    
    for m, l in zip(metrics, lbs):
        for i in range(0, len(axes)):
            ax = axs[i]
            axis = axes[i]
            ax.plot(x_values, m[:, axis], linewidth=2, label=l)
            ax.set(xlabel=xlb[i], ylabel=ylb[i], title=titles[i])
            ax.xaxis.set_ticks(np.linspace(1, total_epochs, num=int(total_epochs/epochs_interval), dtype=np.int32))
            ax.legend(loc='center right')
    
    plt.suptitle(fig_title, fontsize=14, fontweight='bold')
    plt.show()
    
  

def show_image_list(img_list, img_labels, title, cols=2, fig_size=(15, 15), show_ticks=True):
    """
    Utility function to show us a list of traffic sign images
    """
    img_count = len(img_list)
    rows = img_count // cols
    cmap = None

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    
    for i in range(0, img_count):
        img_name = img_labels[i]     
        img = img_list[i]
        if len(img.shape) < 3 or img.shape[-1] < 3:
            cmap = "gray"
            img = np.reshape(img, (img.shape[0], img.shape[1]))
        
        if not show_ticks:            
            axes[i].axis("off")
            
        axes[i].imshow(img, cmap=cmap)
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y = 0.6)
    fig.tight_layout()
    plt.show()
    
    return


def show_random_dataset_images(group_label, imgs, to_show=5):
    """
    This function takes a DataFrame of items group by labels as well as a set of images and randomly selects to_show images to display
    """
    for (lid, lbl), group in group_label:
        #print("[{0}] : {1}".format(lid, lbl))    
        rand_idx = np.random.randint(0, high=group['img_id'].size, size=to_show, dtype='int')
        selected_rows = group.iloc[rand_idx]

        selected_img = list(map(lambda img_id: imgs[img_id], selected_rows['img_id']))
        selected_labels = list(map(lambda label_id: label_id, selected_rows['label_id']))
        show_image_list(selected_img, selected_labels, "{0}: {1}".format(lid, lbl), cols=to_show, fig_size=(7, 7), show_ticks=False)
    
    
########################################################################################################################
######                                        Data manipulation functions                                         ######
########################################################################################################################
    
    
def group_img_id_to_lbl(lbs_ids, lbs_names):    
    """
    Utility function to group images by label 
    """
    arr_map = []
    for i in range(0, lbs_ids.shape[0]):
        label_id = lbs_ids[i]
        label_name = lbs_names[lbs_names["ClassId"] == label_id]["SignName"].values[0]
        arr_map.append({"img_id": i, "label_id": label_id, "label_name": label_name})
    
    return pd.DataFrame(arr_map)


def group_img_id_to_lb_count(img_id_to_lb):
    """
    Returns a pivot table table indexed by label id and label name, where the aggregate function is count
    """
    return pd.pivot_table(img_id_to_lb,index=["label_id","label_name"],values=["img_id"], aggfunc='count')


        

def create_sample_set(grouped_imgs_by_label, imgs, labels, pct=0.4):
    """
    Creates a sample set containing pct elements of the original grouped dataset
    """
    X_sample = []
    y_sample = []

    for (lid, lbl), group in grouped_imgs_by_label:
        group_size = group['img_id'].size
        img_count_to_copy = int(group_size * pct)
        rand_idx = np.random.randint(0, high=group_size, size=img_count_to_copy, dtype='int')

        selected_img_ids = group.iloc[rand_idx]['img_id'].values
        selected_imgs = imgs[selected_img_ids]
        selected_labels = labels[selected_img_ids]
        X_sample = selected_imgs if len(X_sample) == 0 else np.concatenate((selected_imgs, X_sample), axis=0)
        y_sample = selected_labels if len(y_sample) == 0 else np.concatenate((selected_labels, y_sample), axis=0)


    return (X_sample, y_sample)

########################################################################################################################
######                                        Image processing functions                                          ######
########################################################################################################################


def normalise_images(imgs, dist):
    """
    Nornalise the supplied images from data in dist
    """
    std = np.std(dist)
    #std = 128
    mean = np.mean(dist)
    #mean = 128
    return (imgs - mean) / std



def to_grayscale(img):
    """
    Converts an image in RGB format to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def augment_imgs(imgs, p):
    """
    Performs a set of augmentations with with a probability p
    """
    augs =  iaa.SomeOf((1, 2),
          [
              iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 4px (randomly chosen)
              iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
              iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
              iaa.Affine(rotate=(-45, 45)), # rotate by -45 to +45 degrees)
              iaa.Affine(shear=(-10, 10)) # shear by -10 to +10 degrees
          ])

    
    seq = iaa.Sequential([iaa.Sometimes(p, augs)])
    
    return seq.augment_images(imgs)

def augment_imgs_until_n(imgs, n, p):
    """
    Takes care of augmenting images with a probability p, until n augmentations have been created
    """

    i = 0
    aug_imgs = []
    while i < n:
        augs = augment_imgs(imgs, p)
        i += len(augs)
        aug_imgs = augs if len(aug_imgs) == 0 else np.concatenate((aug_imgs, augs))
    
    return aug_imgs[0 : n]


