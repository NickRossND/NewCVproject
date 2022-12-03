import glob
import fnmatch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,  Conv2D, Flatten

path = r'C:\Users\Nick\Downloads\PatientPlots\TestNeuralNet\**\*.npy'
# path = r'G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\CVProject\PatientPlots\**\*.txt'
# print(path)
files = glob.glob(path, recursive=False)
# print(files)
#empty array for training and testing data
X = []
Y = []
#classify files as tumor or not tumor based on location of file
for i in files:
    a = np.load(i)
    X.append(a)
    if fnmatch.fnmatch(i,"*Lesions*"):
        b = 1
    else:
        b = 0
    Y.append(b)
#change vars from list to array
X = np.array(X)
Y = np.array(Y)
Y = to_categorical(Y)
## test to see if patient files and classification match up ##
# print(X[174])
# print(y[174])
# print(X)

## set training and testing data ##
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.90, random_state= 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
#separate some files for validation dataset
x_valid = x_train[-15933:]
y_valid = y_train[-15933:]
x_train = x_train[:-15933]
y_train = y_train[:-15933]

#define input shape
input_shape = (6,6,1)
# define the model
model = Sequential()
model.add(Conv2D(filters=3, kernel_size = (3,3), input_shape = input_shape, activation = 'relu'))
model.add(Conv2D(filters=3, kernel_size = (3,3), input_shape = input_shape, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
epochs = 10
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint("brstTumor.h5", monitor="val_accuracy",verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
model.fit(x_train, y_train, epochs=epochs, verbose=1,  validation_data = (x_valid, y_valid),callbacks=[checkpoint])

## used for testing with test dataset ##
print('-TESTING-----------------------------')
print('Number of test images:', x_test.shape[0])
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# # evaluate predictions and compare predictions to true results
print("Generate predictions for 100 samples")
predictions = model.predict(x_test)

y_test_arg=np.argmax(y_test,axis=1)
Y_pred = np.argmax(predictions,axis=1)
cm = confusion_matrix(y_test_arg, Y_pred)
print('Confusion Matrix')
print(cm)
# print(predictions)
# print(y_test[:100])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot
plt.show()
