import glob
import re
import os
import time
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

def leakyReLu(inputs, alpha=0.05, name=None):
    '''
    Leaky ReLu from Keras's Tensorflow backend implementation
    alpha: slope of negative section.
    '''
    return tf.subtract(tf.nn.relu(inputs), alpha*tf.nn.relu(-inputs), name=name)

def kernel_stride(width, data_format='NCHW'):
    '''
    Create kernel or strides fot the given data format
    Parameters:
    s: the kernel or stride size
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    '''
    if data_format == 'NCHW':
        return [1, 1, width, width]
    else:
        return [1, width, width, 1]

def weights(shape, name, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    '''
    Create a new weight variable
    Parameters:
    shape: shape of the variable
    name: name of the variable
    initializer: initializer for the variable, default is xavier initializer
    '''
    return tf.get_variable(name, shape=shape, initializer=initializer)

def biases(shape, name, initializer=tf.zeros_initializer()):
    '''
    Create a new bias variable
    Parameter:
    shape: shape of the variable
    name: name of the variable
    initializer: initializer for the variable, default is zeros initializer
    '''
    return tf.get_variable(name, shape=shape, initializer=initializer)

def layer_counter():
    '''
    Return the current layer count, this is used to generate unique names
    '''
    if "index" not in layer_counter.__dict__:
        layer_counter.index = 0
    layer_counter.index += 1
    return layer_counter.index

def conv2d(inputs, shape, strides, data_format='NCHW', padding='SAME'):
    '''
    Create a new convolution layer
    Parameters:
    inputs: the input
    shape: shape of the kernel
    strides: the strides
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    padding: the padding
    '''
    layer = layer_counter()
    weight = weights(name='Wconv{}'.format(layer), shape=shape)
    bias = biases(name='Bconv{}'.format(layer), shape=shape[-1:])
    logits = tf.nn.conv2d(inputs, weight, strides=strides, data_format=data_format, padding=padding)
    logits = tf.nn.bias_add(logits, bias, data_format=data_format)
    logits = leakyReLu(logits, alpha=0.02, name='conv{}'.format(layer))
    return logits

def fcnn(inputs, shape, keep_prob=None):
    '''
    Create a fully connection network layer
    Parameters:
    inputs: the inputs layer
    shape: shape of the kernel
    keep_prob: dropout's keep proability. The layer if not have dropout if keep_prob is None
    '''
    layer = layer_counter()
    weight = weights(name='Wfcnn{}'.format(layer), shape=shape)
    bias = biases(name='Bfcnn{}'.format(layer), shape=shape[-1:])
    logits = tf.add(tf.matmul(inputs, weight), bias)
    logits = tf.nn.elu(logits, name='fcn{}'.format(layer))
    if keep_prob != None:
        logits = tf.nn.dropout(logits, keep_prob)
    return logits

def cnn(features, classes, keep_prob=None, data_format='NCHW'):
    '''
    Create our 7-layer CNN model composed of 4 convolutiuon layers and 3 fully connected layers
    Parameters:
    x: the inputs layer
    classes: the number of classes
    keep_prob: dropout's keep proability. The layer if not have dropout if keep_prob is None
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    '''
    kernel_stride2 = kernel_stride(2, data_format=data_format)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x16.
    logits = conv2d(features, shape=(3, 3, 1, 16), strides=[1, 1, 1, 1], data_format=data_format)

    # Layer 2: Convolutional. Input = 32x32x16 output = 32x32x32
    logits = conv2d(logits, shape=(3, 3, 16, 32), strides=[1, 1, 1, 1], data_format=data_format)

    # Layer 3: Convolutional. Input = 32x32x32 output = 32x32x64
    logits = conv2d(logits, shape=(3, 3, 32, 64), strides=[1, 1, 1, 1], data_format=data_format)

    # Pooling. Input = 32x32x64. Output = 16x16x64.
    logits = tf.nn.max_pool(logits, ksize=kernel_stride2, strides=kernel_stride2,
                            data_format=data_format, padding='SAME')

    # Layer 4: Convolutional. Input = 16x16x64, Output = 14x14x128.
    logits = conv2d(logits, shape=(3, 3, 64, 128), strides=[1, 1, 1, 1], data_format=data_format,
                    padding='VALID')

    # Pooling. Input = 14x14x128. Output = 7x7x128.
    logits = tf.nn.max_pool(logits, ksize=kernel_stride2, strides=kernel_stride2,
                            data_format=data_format, padding='SAME')

    # Flatten. Input = 7x7x128. Output = 6272.
    logits = flatten(logits)

    logits = tf.nn.dropout(logits, keep_prob)

    # Layer 5: Fully Connected. Input = 6272. Output = 1600.
    logits = fcnn(logits, shape=(6272, 1600), keep_prob=keep_prob)

    # Layer 6: Fully Connected. Input = 1600. Output = 400.
    logits = fcnn(logits, shape=(1600, 400), keep_prob=keep_prob)

    # Layer 7: Fully Connected. Input = 400. Output = classes, which is 42.
    logits = fcnn(logits, shape=(400, classes))

    return logits

class CNNModel:
    '''
    Encapsulate c CNN model
    '''
    def __init__(self, features, labels, classes, keep_prob, logits):
        '''
        features: the features placeholder
        labels: the labels placeholder
        classes: number of classes
        keep_prob: the placeholder for dropout keep proability
        logits: the tensorflow logitistic model
        '''
        self.features = features
        self.labels = labels
        self.classes = classes
        self.keep_prob = keep_prob
        self.logits = logits

def create_model(feature_shape, classes, feature_dtype=tf.float32, label_dtype=tf.int32):
    '''
    Re-initializeTensorflow environment and create the core network model
    Parameters:
    feature_shape: shape of the features
    classes" classes of the features
    feature_dtype: features's data type, default is float32
    label_dtype: label's data type, default is int32
    '''
    tf.reset_default_graph()
    if feature_shape[0] < feature_shape[1] and feature_shape[0] < feature_shape[2]:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    features = tf.placeholder(feature_dtype, (None,) + feature_shape)
    labels = tf.placeholder(label_dtype, (None))
    keep_prob = tf.placeholder(tf.float32)
    logits = cnn(features, classes=classes, keep_prob=keep_prob, data_format=data_format)
    tf.add_to_collection("features", features)
    tf.add_to_collection("labels", labels)
    tf.add_to_collection("keep_prob", keep_prob)
    tf.add_to_collection("model", logits)
    return CNNModel(features, labels, classes, keep_prob, logits)

def save_current_model(name):
    '''
    Save the current Tensorflow model
    Parameter:
    name: checkpoint name of the model
    '''
    tf.train.export_meta_graph(filename=name+'.meta', clear_devices=True)

def load_model(name, classes):
    '''
    Loade current Tensorflow model
    Parameter:
    name: checkpoint name of the model
    '''
    tf.reset_default_graph()
    tf.train.import_meta_graph(name + '.meta')
    graph = tf.get_default_graph()
    features = graph.get_collection('features')[0]
    labels = graph.get_collection('labels')[0]
    keep_prob = graph.get_collection('keep_prob')[0]
    logits = graph.get_collection('model')[0]
    return CNNModel(features, labels, classes, keep_prob, logits)

def save_trained_model(session, checkpoint, step=None, saver=None):
    '''
    Save the trained model
    Parameters:
    session: the current tensorflow training session
    checkpoint: the checkpoint (the file name prefix where the trained model is saved)
    step: the step (epoch) in which the model was trained
    saver: the saver object to use, a new saver will be used if None is given
    '''
    if saver is None:
        saver = tf.train.Saver(tf.trainable_variables())
    saver.save(session, checkpoint, global_step=step)

def load_trained_model(session, checkpoint, classes):
    '''
    Loade trained model
    Parameters:
    session: the tensorflow session to restore the trained model to
    checkpoint: the checkpoint (the file name prefix where the trained model is to be loaded)
    classes: classes of the model's labels
    '''
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(session, checkpoint)
    graph = tf.get_default_graph()
    features = graph.get_collection('features')[0]
    labels = graph.get_collection('labels')[0]
    keep_prob = graph.get_collection('keep_prob')[0]
    logits = graph.get_collection('model')[0]
    return CNNModel(features, labels, classes, keep_prob, logits)

def find_trained_model(path, checkpoint_prefix):
    '''
    Find the list of checkpoint models, return the checkpoint names of the trained model of format:
    'path/checkpoint_prefix-nn'
    path: path
    checkpoint_prefix: checkpoint name's prefix
    '''
    files = glob.glob('{0}/{1}-*.meta'.format(path, checkpoint_prefix))
    if files is None or len(files) == 0:
        print("No trained model found in {0}/{1}!".format(path, checkpoint_prefix))
        return None
    else:
        # Extract the checkpoint to use
        files.sort(key=os.path.getmtime)
        return [file[:-5] for file in reversed(files)]

def find_tensor(model, name):
    '''
    Find tensor of the given name from current graph. It can then be used to eval inputs,
    or compose a network from it.
    Return: the tensor if found, or (first output, outputs) of an operation if found,
            or None otherwise
    nn: either an instance of CNNMOdel or a tensorflow operation
    name: name of the tensor
    '''
    if isinstance(model, CNNModel):
        model = model.logits

    graph = model.graph

    # First find the operation
    try:
        tensor = graph.get_operation_by_name(name)
    except KeyError: # Unable to find an operation, try to find a tensor
        try:
            tensor = graph.get_tensor_by_name(name)
        except KeyError: # nothing was found
            pass
        else: # Found a tensor
            return tensor
    else:
        if tensor is not None: # we have found an operation
            if len(tensor.outputs) > 1:
                return tensor.outputs[0], tensor.outputs
            else:
                return tensor.outputs[0]

def create_training(model, learning_rate, global_step=None):
    '''
    Create the training model from the core network model. It uses adam optimizer for training
    Parameters:
    logits: the core network model
    labels: the labels
    classes: the number of label classes
    learning_rate: the learning rate
    global_step: the global step, default is None
    '''
    one_hot_y = tf.one_hot(model.labels, model.classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=model.logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)
    return optimizer.minimize(loss_operation, global_step=global_step)

def create_evaluation(model):
    '''
    Create evaluation model from the core model
    Parameters:
    model: the CNN model
    '''
    one_hot_y = tf.one_hot(model.labels, model.classes)
    correct_prediction = tf.equal(tf.argmax(model.logits, 1), tf.argmax(one_hot_y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def create_prediction(model):
    '''
    Create prediction model from the core model
    Parameters:
    model: the CNN model
    '''
    correct_prediction = tf.argmax(model.logits, 1)
    return correct_prediction

def create_softmax_evaluation(model, top=5):
    '''
    Create softmax evaluation model from the core model
    Parameters:
    model: the CNN model
    top: the number of top results to evaluate
    '''
    softmax = tf.nn.softmax(model.logits)
    return softmax, tf.nn.top_k(softmax, top)

def evaluate(session, features, labels, operation, model, batch_size=256):
    '''
    Evaluate the current training model
    Parameters:
    session: the current session
    features: the features to evaluate
    labels: labels of the features
    operation: the evaluation operation
    model: the CNN model
    batch_size: the batch processing size, default is 256
    '''
    num_examples = len(features)
    total_accuracy = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = features[offset:offset+batch_size], labels[offset:offset+batch_size]
        accuracy = session.run(operation, feed_dict={model.features: batch_x, model.labels: batch_y,
                                                     model.keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def train_model(features, labels, valid, y_valid, checkpoint, dropout_keep_prob=0.85, trains=1,
                learning_rate=0.0006, epochs=60, batch_size=256, accept=0.98):
    '''
    This is a generator for generating the best training models. Create and train a CNN model with
    the features, and saves the best training models in checkpoint for prediction operations. The
    generator will yield everytime a best training model is generated.

    Parameters:
    features: the features to train
    labels: labels of the features
    valid: the validation data
    y_valid: the labels of the validation data
    checkpoint: the base name of the checkpoints where the best training models will be saved
    dropout_keep_prob: the keep probability of drop out
    trains: number of trainings to perform for the best model
    learning_rate: thje learning rate
    epochs: the number of epochs for each training
    batch_size: the batch size
    goal: goal of the training accuracy to get
    accept: the minimal accuracy to accept for a training model
    '''
    save_id = 0
    count = 0
    classes = int(max(labels) - min(labels))
    n_samples = len(features)

    model = create_model(features.shape[1:], classes)

    training_operation = create_training(model, learning_rate)
    accuracy_operation = create_evaluation(model)

    while count < trains:
        count += 1
        with tf.Session() as session:
            print("Training {} ...".format(count))
            session.run(tf.global_variables_initializer())
            for i in range(epochs):
                features, labels = shuffle(features, labels)
                for offset in range(0, n_samples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = features[offset:end], labels[offset:end]
                    session.run(training_operation, feed_dict={model.features: batch_x,
                                                               model.labels: batch_y,
                                                               model.keep_prob: dropout_keep_prob})

                validation_accuracy = evaluate(session, valid, y_valid, accuracy_operation, model)
                print("EPOCH {0:3}: Accuracy = {1:.4f}".format(i+1, validation_accuracy))
                if validation_accuracy >= accept:
                    save_id += 1
                    for retry in range(3):
                        try:
                            save_trained_model(session, checkpoint, step=save_id)
                            break
                        except:
                            print("Failed to save trained model!")
                            time.sleep(2)
                    yield validation_accuracy, save_id
