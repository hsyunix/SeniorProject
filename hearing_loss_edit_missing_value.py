from functools import partial

import sys
import tensorflow as tf
import numpy as np
import math
import random
import openpyxl
import os
from tensorflow.contrib.layers import batch_norm, flatten

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)#gpu_id

# loading the data in value type
_, input_file_path = sys.argv
wb = openpyxl.load_workbook(input_file_path, data_only = True)
ws = wb['SSNHL']

# data load
data = []
label = []
metric_seigel = False#output metric

for i in ws.rows:
    col_val = []
    for cell in i:
        col_val.append(cell.value)
        
    if metric_seigel:
        label.append(col_val[-2])
    else:
        label.append(col_val[-1])

    col_val = [col_val[3], col_val[4], col_val[6], col_val[7], col_val[8], col_val[9], col_val[10], col_val[11], col_val[12], col_val[13],\
                col_val[14], col_val[15], col_val[16], col_val[17], col_val[18], \
                col_val[21], col_val[22], col_val[23], col_val[24], col_val[25]]

    data.append(col_val)

# remove the first row 
data = data[1:]
label = label[1:]

def labelClassifier(label):
    for i in range(len(label)):
        if label[i] == 'A':
            label[i] = [1., 0., 0., 0.]
        elif label[i] == 'B':
            label[i] = [0., 1., 0., 0.]
        elif label[i] == 'C':
            label[i] = [0., 0., 1., 0.]
        else:#'D'
            label[i] = [0., 0., 0., 1.]
            
    return label
    
def labelClassifier_Siegel(label):
    for i in range(len(label)):

        if label[i] == 1:
            label[i] = [1., 0., 0., 0.]
        elif label[i] == 2:
            label[i] = [0., 1., 0., 0.]
        elif label[i] == 3:
            label[i] = [0., 0., 1., 0.]
        else:#'4'
            label[i] = [0., 0., 0., 1.]
            
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
        

    return data

if metric_seigel:
    label = labelClassifier_Siegel(label)
else:
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

# normalize the column
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

train_x = np.array(x_train)
train_y = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
train_y = np.argmax(y_train, axis = 1)
valid_y = np.argmax(y_test, axis = 1)

valid_x = np.array(x_test)

def shuffle_batch(inputs, labels, batch_size):
    rnd_idx = np.random.permutation(len(inputs))
    n_batches = len(inputs) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_x, batch_y = inputs[batch_idx], labels[batch_idx]
        yield batch_x, batch_y

################
# layer params #
################
n_inputs = 20
n_hidden1 = 64
n_hidden2 = 128
n_hidden3 = 64
n_outputs = 4
batch_norm_momentum = 0.9
dropout_rate = 0.5

# input layer
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
# output layer
labels = tf.placeholder(tf.int32, shape=[None], name='labels')
# for batch-normalization
training = tf.placeholder_with_default(False, shape=[], name="training")

with tf.name_scope('dnn'):
    # batch normalization layer using partial
    batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training, 
            momentum=batch_norm_momentum)

    # 1st - hidden
    with tf.name_scope('Layer1'):
        hidden1 = tf.layers.dense(inputs, n_hidden1, name="hidden1")
        # batch norm
        bn1 = batch_norm_layer(hidden1)
        # activation function
        bn1_act = tf.nn.relu(bn1)
        # drop out
        bn1_drop = tf.nn.dropout(bn1_act, rate= dropout_rate)
        tf.summary.histogram("Layer1", bn1_drop)

    # 2nd - hidden
    with tf.name_scope('Layer2'):
        hidden2 = tf.layers.dense(bn1_drop, n_hidden2, name="hidden2")
        bn2 = batch_norm_layer(hidden2)
        bn2_act = tf.nn.relu(bn2)
        bn2_drop = tf.nn.dropout(bn2_act, rate= dropout_rate)
        tf.summary.histogram("Layer2", bn2_drop)

    #3rd - hidden
    with tf.name_scope('Layer3'):
        hidden3 = tf.layers.dense(bn2_drop, n_hidden3, name="hidden3")
        bn3 = batch_norm_layer(hidden3)
        bn3_act = tf.nn.relu(bn3)
        bn3_drop = tf.nn.dropout(bn3_act, rate= dropout_rate)

        # outputs
        logits_before_bn = tf.layers.dense(bn3_drop, n_outputs, name="outputs")
        logits = batch_norm_layer(logits_before_bn)
        tf.summary.histogram("Layer3", logits)


with tf.name_scope('Loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    tf.summary.scalar("Loss", loss)


################
# Hyper-params #
################
learning_rate = 0.001
n_epochs = 10
batch_size = 256
iteration = 10000

# moving mean & variance update
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

with tf.name_scope('Accuracy'):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)



with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/dolbal")
    writer.add_graph(sess.graph)
    tf.global_variables_initializer().run()
    
    for epoch in range(n_epochs):
        for idx, (batch_x, batch_y) in enumerate(shuffle_batch(train_x, train_y, batch_size)):
            for step in range(iteration):

                cost_val, summary, _ = sess.run([loss, merged_summary, train_op], feed_dict = {inputs: batch_x, labels: batch_y, training: True})
                writer.add_summary(summary, global_step=step)

                # validation
                if (step + 1) % 1000 == 0:
                    accuracy_val = accuracy.eval(feed_dict={inputs: valid_x, labels: valid_y})
                    print('epoch: {:03d}, batch_idx:{:03d} iteration: {:05d} loss: {:.8f} valid. Acc: {:.4f}'.format(epoch + 1, idx + 1, step + 1, cost_val, accuracy_val))

