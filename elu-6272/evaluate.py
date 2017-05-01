import glob
import re
import numpy as np
import cv2
import tensorflow as tf
import utils
import model

import config

files = glob.glob('../examples/traff-*.jpg')
samples = np.array([cv2.cvtColor(cv2.imread(file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for file in files])
labels = [int(re.findall(r'\d+', file)[0]) for file in files]
images = utils.normal_gray(samples)
plt = utils.rand_visual([samples, images], rows=2, columns=5, indices=[[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])
plt.ion()
plt.show()
images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))
n_classes = 42

print("Samples: ", images.shape[0])
print("Labels: ", labels)
print("Classes: ", n_classes)

for trained in model.find_trained_model(config.CHECKPOINT_FOLDER, config.CHECKPOINT_NAME):
    tf.reset_default_graph()
    with tf.Session() as session:
        # Load the pre-trained model
        cnn = model.load_trained_model(session, trained, n_classes)
        # Perdiction
        accuracy_operation = model.create_evaluation(cnn)
        prediction_operation = model.create_prediction(cnn)
        softmax_operation = model.create_softmax_evaluation(cnn, 3)
        # Evaluate the test data against the pre-trained data
        prediction = session.run(prediction_operation, feed_dict={cnn.features: images,
                                                                  cnn.labels: labels,
                                                                  cnn.keep_prob:1.0})
        # Create the accuracy perdiction
        test_accuracy = model.evaluate(session, images, labels, accuracy_operation, cnn)
        # Evaluate the softmax performance
        softmax, softmax_performance = session.run(softmax_operation,
                                                   feed_dict={cnn.features: images,
                                                              cnn.labels: labels,
                                                              cnn.keep_prob:1.0})
        print("Prediction:\n", prediction)
        print("\tTest {1}, accuracy = {0:.4f}%".format(test_accuracy*100, trained))
        print("Performance:\n", softmax_performance.values)
