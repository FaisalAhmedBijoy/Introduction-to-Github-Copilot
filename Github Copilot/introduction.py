# check a number that is even or odd
number = int(input("Enter a number: "))
if number % 2 == 0:
    print("The number is even") 
else:
    print("The number is odd")  # end of if statement

# check if a number is positive or negative
number = int(input("Enter a number: "))
if number > 0:
    print("The number is positive")
elif number < 0:
    print("The number is negative")
else:
    print("The number is zero")

# print 1 to 100 using loop
for i in range(1, 101):
    print(i)    
#print 1 to 100 using while loop
i = 1
while i <= 100:
    print(i)
    i += 1
# print fibonacci series
a, b = 0, 1
while b < 50:
    print(b)
    a, b = b, a + b

# print a triangle
for i in range(1, 11):
    for j in range(1, i + 1):
        print(j, end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j > 10:
            print("*", end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j < 11:
            print("*", end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j < 11:
            print("*", end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j < 11:
            print("*", end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j < 11:
            print("*", end=" ")
    print("")
# print a diamond shape
for i in range(1, 11):
    for j in range(1, 11):
        if i + j < 11:
            print("*", end=" ")
    print("")

# find a value using binary search
def binary_search(list, item):
    low = 0
    high = len(list) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None
# multiple of 3 and 5
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
# multiplication of two matrices
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(len(matrix1)):
    for j in range(len(matrix2[0])):
        for k in range(len(matrix2)):
            result[i][j] += matrix1[i][k] * matrix2[k][j]
for r in result:
    print(r)
# multiplication of two matrices
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(len(matrix1)):
    for j in range(len(matrix2[0])):
        for k in range(len(matrix2)):
            result[i][j] += matrix1[i][k] * matrix2[k][j]
for r in result:
    print(r)



# image classification model using CNN in Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,  
                                        zoom_range=0.2,
                                        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(150, 150), 
                                                    batch_size=32,
                                                    class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='binary')
model.fit_generator(training_set,
                        steps_per_epoch=8000,   
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
# object detection using tensorflow
import tensorflow as tf
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
model = tf.keras.models.load_model('model.h5')
model.summary()
def get_images_and_labels(path):
    images = []
    labels = []
    for f in listdir(path):
        if isfile(join(path, f)):
            images.append(cv2.imread(join(path, f)))
            labels.append(int(f.split('.')[0]))
    return np.array(images), np.array(labels, dtype=np.int32)
images, labels = get_images_and_labels('dataset/test_set')
predictions = model.predict(images)
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        print(labels[i], ':', 'dog')
    else:
        print(labels[i], ':', 'cat')
# implement support vector machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

model = SVC(kernel='linear', C=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))



# check a number even or odd
def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False
print(is_even(5))

# implement bfs algorithm
def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited
graph = {'A': set(['B', 'C']),

