import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 256
learning_rate_init = 0.0025
decay_steps = 1000
decay_rate = 0.8
drop_out_rate = 0.5

stable_gap = 0.002
quit_stabled_times = 5

global_step = tf.Variable(0, trainable=False)
learning_rate_dacayed = tf.compat.v1.train.exponential_decay(learning_rate_init, global_step, decay_steps, decay_rate,
                                                             staircase=False)


def lenet_tf(X_input, keep_prob, drop_out=True):
    """
    Builds up a tensorflow NN net with the given input, the structure of the model is stated inside file "writeup_lix.md".
    :param X_input: A data input tensor
    """
    mu = 0
    sigma = 0.1

    ###### Part 2: Configuration parameters for all layers. #####

    w_dic = {  # The dictionary for all weights variables & sizes
        "conv1": tf.Variable(tf.truncated_normal((5, 5, 1, 6), mean=mu, stddev=sigma), name="w_conv1"),
        "pool1": [1, 2, 2, 1],
        "conv2": tf.Variable(tf.truncated_normal((5, 5, 6, 16), mean=mu, stddev=sigma), name="w_conv2"),
        "pool2": [1, 2, 2, 1],
        "linear1": tf.Variable(tf.truncated_normal((400, 120), mean=mu, stddev=sigma), name="w1"),
        "linear2": tf.Variable(tf.truncated_normal((120, 84), mean=mu, stddev=sigma), name="w2"),
        "linear3": tf.Variable(tf.truncated_normal((84, 43), mean=mu, stddev=sigma), name="w3")
    }

    b_dic = {  # The dictionary for all biases
        "conv1": tf.Variable(tf.zeros((6)), name="b_conv1"),
        "conv2": tf.Variable(tf.zeros((16)), name="b_conv2"),
        "linear1": tf.Variable(tf.zeros((120)), name="b1"),
        "linear2": tf.Variable(tf.zeros((84)), name="b2"),
        "linear3": tf.Variable(tf.zeros((43)), name="b3")
    }

    s_dic = {  # The dictionary for all strides
        "conv1": [1, 1, 1, 1],
        "pool1": [1, 2, 2, 1],
        "conv2": [1, 1, 1, 1],
        "pool2": [1, 2, 2, 1]
    }

    p_dic = {  # The dictionary for all padding types
        "conv1": "VALID",
        "pool1": "VALID",
        "conv2": "VALID",
        "pool2": "VALID"
    }

    ##### Part 3: The tensorflow Neuronetwork structure

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_output = tf.nn.bias_add(tf.nn.conv2d(X_input, w_dic["conv1"], strides=s_dic["conv1"], padding=p_dic["conv1"]),
                                  b_dic["conv1"])

    # Activation.
    conv1_activated = tf.nn.relu(conv1_output)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1_output = tf.nn.max_pool(conv1_activated, w_dic["pool1"], s_dic["pool1"], p_dic["pool1"])

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_output = tf.nn.bias_add(
        tf.nn.conv2d(pool1_output, w_dic["conv2"], strides=s_dic["conv2"], padding=p_dic["conv2"]), b_dic["conv2"])

    # Activation.
    conv2_activated = tf.nn.relu(conv2_output)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2_output = tf.nn.max_pool(conv2_activated, w_dic["pool2"], s_dic["pool2"], p_dic["pool2"])

    # Flatten. Input = 5x5x16. Output = 400.
    flatten_output = tf.contrib.layers.flatten(pool2_output)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    linear1_output = tf.add(tf.matmul(flatten_output, w_dic["linear1"]), b_dic["linear1"])
    if drop_out:
        linear1_output = tf.nn.dropout(linear1_output, keep_prob)

    # Activation.
    linear1_activated = tf.nn.relu(linear1_output)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    linear2_output = tf.add(tf.matmul(linear1_activated, w_dic["linear2"]), b_dic["linear2"])
    if drop_out:
        linear2_output = tf.nn.dropout(linear2_output, keep_prob)

    # Activation.
    linear2_activated = tf.nn.relu(linear2_output)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(linear2_activated, w_dic["linear3"]), b_dic["linear3"])

    return logits
