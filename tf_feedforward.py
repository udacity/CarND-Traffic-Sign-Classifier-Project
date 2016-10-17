"""
Start pure TensorFlow version.
"""
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

with open('train.p', mode='rb') as f:
    train = pickle.load(f)
with open('test.p', mode='rb') as f:
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
# number of traffic signs
n_classes = 43
epochs = 10
input_shape = X_train.shape[1:]

feature_size = 32*32*3

# tf.nn.conv2d(input, filter, strides, padding)
W1 = tf.Variable(tf.random_normal((feature_size, 100), mean=0, stddev=0.01) , name="W1")
W2 = tf.Variable(tf.random_normal((100, n_classes), mean=0, stddev=0.01) , name="W2")

b1 = tf.Variable(tf.zeros((100)) , name="b1")
b2 = tf.Variable(tf.zeros((n_classes)) , name="b2")

features = tf.placeholder(tf.float32, (None, feature_size), name="features")
labels = tf.placeholder(tf.int64, (None), name="labels")

out1 = tf.matmul(features, W1) + b1
out1 = tf.nn.relu(out1)
logits = tf.matmul(out1, W2) + b2

correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels), 0)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(epochs):
		for offset in range(0, X_train.shape[0], batch_size):
			end = offset + batch_size
			batch_X = X_train[offset:end]
			batch_y = y_train[offset:end]	
			bs = batch_X.shape[0]
			batch_X = batch_X.reshape(bs, feature_size)

			l, _ = sess.run([loss, train_op], feed_dict={features: batch_X, labels: batch_y})

		val_l, val_acc = sess.run([loss, accuracy], feed_dict={features: X_val.reshape(X_val.shape[0], feature_size), labels: y_val})
		print("Validation	Loss =", val_l)
		print("Validation	Accuracy =", val_acc)

	
	test_l, test_acc = sess.run([loss, accuracy], feed_dict={features: X_test.reshape(X_test.shape[0], feature_size), labels: y_test})
	print("Testing	Loss =", test_l)
	print("Testing	Accuracy =", test_acc)


