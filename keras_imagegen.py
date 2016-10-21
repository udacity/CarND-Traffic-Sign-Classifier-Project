"""
Generates additional data with ImageDataGenerator
"""
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Dense, ELU, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
with open('./data/test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

batch_size = 64
# number of traffic signs
n_classes = 43
nb_epoch = 100
input_shape = X_train.shape[1:]

model = Sequential()
model.add(Lambda(lambda x: x/255, input_shape=input_shape, output_shape=input_shape))
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

train_datagen = ImageDataGenerator()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20
)

model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    samples_per_epoch=X_train.shape[0],
    nb_epoch=nb_epoch,
    validation_data=(X_val, y_val)
)

# Note: before you load the model with keras.models.load_model
# you have to call keras.backend.set_learning_phase to either:
# 1 (if you want to train more)
# 0 (if you want to test)

model.save('imagegen.h5')
_, acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing accuracy =", acc)
