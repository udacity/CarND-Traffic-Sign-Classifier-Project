import glob
import math
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import model

image_file = '../examples/traff-23.jpg'
trained = 'traffic_signs-122'
n_classes = 42

def show_feature_map(session, cnn, tensors, image, activation_min=-1, activation_max=-1):
    # image: the test image being fed into the network to produce the feature maps
    # tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
    # activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
    # plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    plt_num = 0
    for name in tensors:
        tensor = model.find_tensor(cnn, name)
        if tensor is not None:
            plt_num += 1
            activation = tensor.eval(session=session, feed_dict={cnn.features : [image,]})
            print(activation.shape)
            featuremaps = activation.shape[1]
            plt.figure(plt_num, figsize=(15,15))
            for featuremap in range(featuremaps):
                plt.subplot(int(math.ceil(featuremaps / 8.0)), 8, featuremap+1) # sets the number of feature maps to show on each row and column
                plt.title(str(featuremap)) # displays the feature map number
                if activation_min != -1 & activation_max != -1:
                    plt.imshow(activation[0, featuremap, :,:], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
                elif activation_max != -1:
                    plt.imshow(activation[0, featuremap, :,:], interpolation="nearest", vmax=activation_max, cmap="gray")
                elif activation_min !=-1:
                    plt.imshow(activation[0, featuremap, :,:], interpolation="nearest", vmin=activation_min, cmap="gray")
                else:
                    plt.imshow(activation[0, featuremap, :,:], interpolation="nearest", cmap="gray")

image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
image = utils.normal_gray(np.array([image,]))[0]
image = image.reshape((1, image.shape[0], image.shape[1]))
with tf.Session() as session:
    cnn = model.load_trained_model(session, trained, n_classes)
    show_feature_map(session, cnn, ['conv1', 'conv2', 'conv3', 'conv4'], image)

plt.show()