import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense,Dropout,BatchNormalization
from HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

INPUT_SHAPE=103
NUM_CLASSES=10
model = Sequential([
    Dense(128, activation='relu', input_shape=(INPUT_SHAPE,)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(NUM_CLASSES,activation='softmax'),
])

LEARNING_RATE=0.00003
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy',])
print(model.summary())

loader=HyperDataLoader()
X,y=loader.generate_vectors("PaviaU")
y=to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


history = model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1)
print(history)
results = model.evaluate(X_test, y_test, batch_size=256)
print("Accuracy over test set is {0}".format(results))

