"""
This gets ~0.97-0.98 accuracy

Using a very aggresive dropout of 0.8.
"""
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Dense, ELU, BatchNormalization

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

batch_size = 32
# number of traffic signs
n_classes = 43
nb_epoch = 10
input_shape = X_train.shape[1:]

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(Dropout(.8))
model.add(ELU())
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    verbose=1,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    validation_data=(X_val, y_val))

# Note: before you load the model with keras.models.load_model
# you have to call keras.backend.set_learning_phase to either:
# 1 (if you want to train more)
# 0 (if you want to test)

model.save('models/convnet.h5')
_, acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing accuracy =", acc)
