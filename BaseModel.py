import numpy as np
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

LEARNING_RATE=0.00002
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy',])
print(model.summary())



loader=HyperDataLoader()
X,y=loader.generate_vectors("PaviaU")
y=to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

dummy_x=X_train[0].reshape(1,103)
print("Xtrain pred:",model.predict(dummy_x))
v_i=np.zeros((1,103))
v_zeros=np.zeros((1,103))
pred_zero = model.predict(v_zeros)
v_i[0][10] = 1
pred_i10 = model.predict(v_i)
diff_pred = pred_i10-pred_zero
print("pred_zero",pred_zero,"argmax",np.argmax(pred_zero))
print("pred_i10",pred_i10,"argmax",np.argmax(pred_i10))
print("diff_pred",diff_pred,"argmax",np.argmax(diff_pred))

history = model.fit(X_train, y_train, epochs=4, batch_size=256, verbose=1)
#print(history)
results = model.evaluate(X_test, y_test, batch_size=256)
print("Accuracy over test set is {0}".format(results))

pred_zero = model.predict(v_zeros)
pred_i10 = model.predict(v_i)
diff_pred = pred_i10-pred_zero
print("pred_zero",pred_zero,"argmax",np.argmax(pred_zero))
print("pred_i10",pred_i10,"argmax",np.argmax(pred_i10))
print("diff_pred",diff_pred,"argmax",np.argmax(diff_pred))

heatmaps=[]
def build_class_heatmap(model:keras.Model,data,lables,features_num,class_label):
    arr=np.zeros(features_num)
    for x,y in zip(data,lables):
        if y == class_label:
            for i in range(features_num):
                v_i = np.zeros(features_num)
                v_i[i] = x[i]
                pred_i = model.predict(v_i)
                pred_zero = model.predict(np.zeros(features_num))
                diff = pred_i-pred_zero
                arr[i] += (diff[i]-diff[np.argmax(diff)])
    return arr


