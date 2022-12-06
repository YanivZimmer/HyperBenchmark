import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense,Dropout,BatchNormalization
from HyperDataLoader import HyperDataLoader

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
    Dense(NUM_CLASSES,activation='sigmoid'),
])

print("Hellowwww")
LEARNING_RATE=0.00003
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy',])
print("Hello")
loader=HyperDataLoader()
X,Y=loader.generate_vectors("PaviaU")
print(X.shape)
print(X[0].shape)
print(Y.shape)
print(Y[0].shape)
print(model.predict(X[0]).shape,Y[0].shape)
history = model.fit(X, Y, epochs=2, batch_size=4,verbose=1)

print(history)
