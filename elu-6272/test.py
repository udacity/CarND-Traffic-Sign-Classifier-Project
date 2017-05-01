import pickle
import time
import tensorflow as tf
import utils
import model

import config

with open(config.TESTING_FILE, mode='rb') as f:
    test_data = pickle.load(f)

X_test, y_test = test_data['features'], test_data['labels']

# How many unique classes/labels there are in the dataset.
CLASSES = 42

print("Number of testing examples =", len(X_test))
print("Image data shape =", X_test.shape[1:])
print("Number of classes =", CLASSES)

def test(features, labels, checkpoint_folder='.', checkpoint_name='trained'):
    '''
	Test the trained models
	features: the features
	labels: the labels
	checkpoint_folder: the checkpoint's folder
	checkpoint_name: name of the checkpoint
	'''
	# Reshape the test data for NCHW format
    features = features.reshape((features.shape[0], 1, features.shape[1], features.shape[2]))

    start_time = time.time()
    for trained in model.find_trained_model(checkpoint_folder, checkpoint_name):
		# Initialize the Tensorflow environment
        tf.reset_default_graph()
        with tf.Session() as session:
			# Load the pre-trained model
            cnn = model.load_trained_model(session, trained, CLASSES)
            print("Trained model: {}".format(trained))
            # Create the accuracy perdiction
            accuracy_operation = model.create_evaluation(cnn)
			# Evaluate the test data against the pre-trained data
            test_accuracy = model.evaluate(session, features, labels, accuracy_operation, cnn)
            print("\tTest accuracy for {0} samples = {1:.3f}".format(len(features), test_accuracy))
    print("\tTest time {:.3} seconds".format((time.time() - start_time)))

test(utils.normal_gray(X_test), y_test, config.CHECKPOINT_FOLDER, config.CHECKPOINT_NAME)
