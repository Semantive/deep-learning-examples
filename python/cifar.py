import cPickle as pickle
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_data(path):
    x_train = np.zeros((50000, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((50000,), dtype="uint8")

    for i in range(1, 6):
        data = unpickle(os.path.join(path, 'data_batch_' + str(i)))
        images = data['data'].reshape(10000, 3, 32, 32)
        labels = data['labels']
        x_train[(i - 1) * 10000:i * 10000, :, :, :] = images
        y_train[(i - 1) * 10000:i * 10000] = labels

    test_data = unpickle(os.path.join(path, 'test_batch'))
    x_test = test_data['data'].reshape(10000, 3, 32, 32)
    y_test = np.array(test_data['labels'])

    return x_train, y_train, x_test, y_test


def unpickle(file):
    f = open(file, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict


net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dense', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    input_shape=(None, 3, 32, 32),
    conv2d1_num_filters=20,
    conv2d1_filter_size=(5, 5),
    conv2d1_stride=(1, 1),
    conv2d1_pad=(2, 2),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool1_pool_size=(2, 2),
    conv2d2_num_filters=20,
    conv2d2_filter_size=(5, 5),
    conv2d2_stride=(1, 1),
    conv2d2_pad=(2, 2),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool2_pool_size=(2, 2),
    dense_num_units=1000,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    update=nesterov_momentum,
    update_momentum=0.9,
    update_learning_rate=0.0001,
    max_epochs=100,
    verbose=True
)

x_train, y_train, x_test, y_test = load_data(os.path.expanduser('~/deep-learning/data/cifar-10-batches-py'))

network = net.fit(x_train, y_train)
predictions = network.predict(x_test)

print classification_report(y_test, predictions)
print accuracy_score(y_test, predictions)
