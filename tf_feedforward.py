"""
TensorFlow feedforward net on German Traffic Sign Dataset
"""
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

with open('data/train.p', mode='rb') as f:
    train = pickle.load(f)
with open('data/test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# 0-255 -> 0-1
X_train /= 255
X_val /= 255
X_test /= 255

batch_size = 64
n_classes = 43 # number of traffic signs
epochs = 10
input_shape = X_train.shape[1:]
feature_size = np.prod(input_shape)
hidden_size = 100

features = tf.placeholder(tf.float32, (None,) + input_shape, name="features")
labels = tf.placeholder(tf.int64, (None), name="labels")

# 4D -> 2D
flattened = tf.reshape(features, [-1, feature_size])

W1 = tf.Variable(tf.random_normal((feature_size, hidden_size), mean=0, stddev=0.01) , name="W1")
W2 = tf.Variable(tf.random_normal((hidden_size, n_classes), mean=0, stddev=0.01) , name="W2")
b1 = tf.Variable(tf.zeros((hidden_size)) , name="b1")
b2 = tf.Variable(tf.zeros((n_classes)) , name="b2")

out1 = tf.matmul(flattened, W1) + b1
out1 = tf.nn.relu(out1)
logits = tf.matmul(out1, W2) + b2

correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels), 0)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(epochs):
		for offset in range(0, X_train.shape[0], batch_size):
			end = offset + batch_size
			X_batch = X_train[offset:end]
			y_batch = y_train[offset:end]	

			sess.run([loss_op, train_op], feed_dict={features: X_batch, labels: y_batch})

		val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={features: X_val, labels: y_val})
		print("Validation	Loss =", val_loss)
		print("Validation	Accuracy =", val_acc)
		
	test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={features: X_test, labels: y_test})
	print("Testing	Loss =", test_loss)
	print("Testing	Accuracy =", test_acc)


