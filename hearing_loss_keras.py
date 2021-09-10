import sys
import tensorflow as tf
import numpy as np
import math
import random
import openpyxl

from tensorflow.contrib.layers import batch_norm, flatten

# loading the data in value type
_, input_file_path = sys.argv
wb = openpyxl.load_workbook(input_file_path, data_only = True)
ws = wb['SSNHL']

# data load
data = []
label = []
for i in ws.rows:
    row_val = []
    for cell in i:
        row_val.append(cell.value)
    label.append(row_val[-1])
    row_val = row_val[3:-1]
    row_val[2] = row_val[2][:8] # slicing 'SSNHL[]...' to 'SSNHL[]'
    data.append(row_val)

# eliminate the attribute name
data = data[1:]
label = label[1:]

# label classification function. A -> [1.0, 0.0, 0.0], B, C -> [0.0, 1.0, 0.0], D -> [0.0, 0.0, 1.0]
def labelClassifier(label):
    for i in range(len(label)):
        if label[i] == 'A':
            label[i] = [1., 0., 0.]
        elif label[i] == 'D':
            label[i] = [0., 0., 1.]
        else:
            label[i] = [0., 1., 0.]
    return label

# data pre-processing function
def preProcessing(data):
    for i in range(len(data)):
        #normalize the age
        data[i][1] = int(data[i][1]) / 100

        # if male, replace 'M' with 1
        if data[i][0] == 'M':
            data[i][0] = 1
        else: data[i][0] = 0    # if female, replace 'F' with 0

        # if right SSNHL, replace SSNHL[R] with 1.0
        if '[R]' in data[i][2]:
            data[i][2] = 1.
        elif '[L]' in data[i][2]:   # if left SSNHL, replace SSNHL[L] with 0.0
            data[i][2] = 0.
        else: data[i][2] = .5   # if binary, replace SSNHL[B] with 0.5
    return data

label = labelClassifier(label)
data = preProcessing(data)

# handling the missing data
def handle_missingData(data):
    # calculate all the avg.
    avg_list = []
    for i in range(len(data[0])):
        sum = 0
        for j in range(len(data)):
            if isinstance(data[j][i], int) or isinstance(data[j][i], float):
                sum += data[j][i]
        avg_list.append(sum / len(data))

    # fill the missing field with the avg. value
    for i in range(len(data[0])):
        for j in range(len(data)):
            if not ((isinstance(data[j][i], int) or isinstance(data[j][i], float))):
                data[j][i] = avg_list[i]
    return data
data = handle_missingData(data)

# data type unifying
data = np.array(data, np.float32).T
label = np.array(label, np.float32)

#normalize the column
for i in range(len(data)):
    if min(data[i]) < 0 or max(data[i]) > 1:
        data[i] = data[i] / np.sqrt(np.sum(data[i] ** 2))
data = data.T

random_sample = random.sample(range(len(data)), int(len(data) * 0.85))

x_train = []
y_train = []
x_test = []
y_test = []

for idx, datum in enumerate(data):
    if idx in random_sample:
        x_train.append(datum)
        y_train.append(label[idx])
    else:
        x_test.append(datum)
        y_test.append(label[idx])

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = np.argmax(y_train, axis = 1)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = np.argmax(y_test, axis = 1)

model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(512, activation=tf.sigmoid),
    # tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)