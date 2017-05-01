import pickle
import time
import utils
import model
import config

EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DROPOUT_KEEP = 0.5
TRAINING_GOAL = 0.999
ACCEPT = 0.993
TRAINS = 3

with open(config.TRAINING_FILE, mode='rb') as f:
    train_data = pickle.load(f)
with open(config.VALIDATION_FILE, mode='rb') as f:
    validation_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_valid, y_valid = validation_data['features'], validation_data['labels']

# How many unique classes/labels there are in the dataset.
classes = max(y_train) - min(y_train)

print("Number of training examples =", len(X_train))
print("Number of validation examples =", len(X_valid))
print("Image data shape =", X_train.shape[1:])
print("Number of classes =", classes)

def train(samples, labels, valid_samples, valid_labels, trains, checkpoint):
    '''
	Train model
	samples: the training samples
	labels: the training labels
	valid_samples: the validation samples
	valid_labels: the validation labels
	trains: the number of trainings to perform
	checkpoints: the checkpoints for trained model
	'''
    # Reshape the training and validation data for NCHW format
    samples = samples.reshape((samples.shape[0], 1, samples.shape[1], samples.shape[2]))
    valid_samples = valid_samples.reshape((valid_samples.shape[0], 1, valid_samples.shape[1],
                                           valid_samples.shape[2]))
    start_time = time.time()
    # Train the model with the training and validation data
    for accuracy, best_index in model.train_model(samples, labels, valid_samples, valid_labels,
                                                  dropout_keep_prob=DROPOUT_KEEP,
                                                  trains=trains, checkpoint=checkpoint,
                                                  learning_rate=LEARNING_RATE, epochs=EPOCHS,
                                                  batch_size=BATCH_SIZE, accept=ACCEPT):
        print("Save model: {0}-{1}, accuracy: {2:.4f}".format(checkpoint, best_index, accuracy))
        if accuracy >= TRAINING_GOAL:
            break

    print("Total training time: {:.3} seconds".format((time.time() - start_time)))

train(utils.normal_gray(X_train), y_train, utils.normal_gray(X_valid), y_valid, TRAINS,
      config.CHECKPOINT)
