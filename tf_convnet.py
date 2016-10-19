"""
TensorFlow convolutional net on German Traffic Sign Dataset
"""
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

with open('data/train.p', mode='rb') as f:
    train = pickle.load(f)
with open('data/test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(
    train['features'], train['labels'], test_size=0.33, random_state=0)
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

batch_size = 64
n_classes = 43 # number of traffic signs
epochs = 10
input_shape = X_train.shape[1:]
print(input_shape)

# 0-255 -> 0-1
X_train /= 255
X_val /= 255
X_test /= 255


def conv2d(input, filter, strides=(1, 1), padding='VALID'):
    kernel = tf.Variable(tf.random_normal(filter, mean=0.0, stddev=0.01))
    bias = tf.Variable(tf.zeros((filter[-1])))
    strides = (1,) + strides + (1,)
    return tf.nn.conv2d(input, kernel, strides, padding) + bias


def maxpool2d(input, ksize, strides=(1, 1), padding='VALID'):
    strides = (1,) + strides + (1,)
    ksize = (1,) + ksize + (1,)
    return tf.nn.max_pool(input, ksize, strides, padding)

# placeholders
features = tf.placeholder(tf.float32, (None,) + input_shape, name="features")
labels = tf.placeholder(tf.int64, (None), name="labels")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# # 32 3x3 filters stride 1x1
with tf.variable_scope('conv1'):
    conv1 = conv2d(features, [3, 3, 3, 32])
    conv1 = tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1, (2, 2))
# # 32 3x3 filters stride 1x1
with tf.variable_scope('conv2'):
    conv2 = conv2d(conv1, [3, 3, 32, 32])
    conv2 = tf.nn.relu(conv2)
    conv2 = maxpool2d(conv2, (2, 2))
# # 64 3x3 filters stride 1x1
with tf.variable_scope('conv3'):
    conv3 = conv2d(conv2, [3, 3, 32, 64])
    conv3 = tf.nn.relu(conv3)
    conv3 = maxpool2d(conv3, (2, 2))

# Flatten
conv3_shape = conv3.get_shape().as_list()
flattened = tf.reshape(conv3, [-1, conv3_shape[1] * conv3_shape[2] * conv3_shape[3]])
flattened_shape = flattened.get_shape().as_list()

# FC1
fc_W1 = tf.Variable(tf.random_normal((flattened_shape[-1], 512), mean=0, stddev=0.01), name="fc_W1")
fc_b1 = tf.Variable(tf.zeros((512)), name="fc_b1")

# FC2
fc_W2 = tf.Variable(tf.random_normal((512, n_classes), mean=0, stddev=0.01), name="fc_W2")
fc_b2 = tf.Variable(tf.zeros((n_classes)), name="fc_b2")

fc1 = tf.matmul(flattened, fc_W1) + fc_b1
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)
logits = tf.matmul(fc1, fc_W2) + fc_b2

correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_op= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels), 0)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)


def eval_on_data(X, y, sess):
    n = 0
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy], feed_dict={features: X_batch, labels: y_batch, keep_prob: 1.0})
        n += 1
        total_loss += loss
        total_acc += acc

    return total_loss/n, total_acc/n


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            X_batch = X_train[offset:end]
            y_batch = y_train[offset:end]

            sess.run([loss_op, train_op], feed_dict={features: X_batch, labels: y_batch, keep_prob: 0.5})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

    test_loss, test_acc = eval_on_data(X_test, y_test, sess)
    print("Testing Loss =", test_loss)
    print("Testing Accuracy =", test_acc)

