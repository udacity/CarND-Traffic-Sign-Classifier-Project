import os

CHECKPOINT_FOLDER = os.path.expanduser('~') + '/projects/CarND/trained/'
CHECKPOINT_NAME = 'traffic_signs'
TRAINING_FILE = '../data/train.p'
VALIDATION_FILE = '../data/valid.p'
TESTING_FILE = '../data/test.p'

CHECKPOINT = CHECKPOINT_FOLDER + CHECKPOINT_NAME
