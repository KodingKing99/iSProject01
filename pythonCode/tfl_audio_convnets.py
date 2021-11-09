#############################################################
# module: tfl_audio_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for audio ConvNets for Project 1
# to install tflearn to go http://tflearn.org/installation/
#############################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

## Paths to all datasets. Change accordingly.
base_path = '/home/nicksorenson/School/intellSys/project01/datasets/tflearn/'
NETPATH = '/home/nicksorenson/School/intellSys/project01/brains/cnAudioBrains/'

BUZZ1_base_path = base_path + 'BUZZ1/'
BUZZ2_base_path = base_path + 'BUZZ2/'
BUZZ3_base_path = base_path + 'BUZZ3/'

## let's load BUZZ1
base_path = BUZZ1_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ1_train_X = load(base_path + 'train_X.pck')
BUZZ1_train_Y = load(base_path + 'train_Y.pck')
BUZZ1_test_X = load(base_path + 'test_X.pck')
BUZZ1_test_Y = load(base_path + 'test_Y.pck')
BUZZ1_valid_X = load(base_path + 'valid_X.pck')
BUZZ1_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ1_train_X.shape)
print(BUZZ1_train_Y.shape)
print(BUZZ1_test_X.shape)
print(BUZZ1_test_Y.shape)
print(BUZZ1_valid_X.shape)
print(BUZZ1_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ1_train_X = BUZZ1_train_X.reshape([-1, 4000, 1, 1])
BUZZ1_test_X  = BUZZ1_test_X.reshape([-1, 4000, 1, 1])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BUZZ1_train_X.shape[0] == BUZZ1_train_Y.shape[0]
assert BUZZ1_test_X.shape[0]  == BUZZ1_test_Y.shape[0]
assert BUZZ1_valid_X.shape[0] == BUZZ1_valid_Y.shape[0]

## let's load BUZZ2
base_path = BUZZ2_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ2_train_X = load(base_path + 'train_X.pck')
BUZZ2_train_Y = load(base_path + 'train_Y.pck')
BUZZ2_test_X = load(base_path + 'test_X.pck')
BUZZ2_test_Y = load(base_path + 'test_Y.pck')
BUZZ2_valid_X = load(base_path + 'valid_X.pck')
BUZZ2_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ2_train_X.shape)
print(BUZZ2_train_Y.shape)
print(BUZZ2_test_X.shape)
print(BUZZ2_test_Y.shape)
print(BUZZ2_valid_X.shape)
print(BUZZ2_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ2_train_X = BUZZ2_train_X.reshape([-1, 4000, 1, 1])
BUZZ2_test_X = BUZZ2_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ2_train_X.shape[0] == BUZZ2_train_Y.shape[0]
assert BUZZ2_test_X.shape[0]  == BUZZ2_test_Y.shape[0]
assert BUZZ2_valid_X.shape[0] == BUZZ2_valid_Y.shape[0]

## let's load BUZZ3
base_path = BUZZ3_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ3_train_X = load(base_path + 'train_X.pck')
BUZZ3_train_Y = load(base_path + 'train_Y.pck')
BUZZ3_test_X = load(base_path + 'test_X.pck')
BUZZ3_test_Y = load(base_path + 'test_Y.pck')
BUZZ3_valid_X = load(base_path + 'valid_X.pck')
BUZZ3_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ3_train_X.shape)
print(BUZZ3_train_Y.shape)
print(BUZZ3_test_X.shape)
print(BUZZ3_test_Y.shape)
print(BUZZ3_valid_X.shape)
print(BUZZ3_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ3_train_X = BUZZ3_train_X.reshape([-1, 4000, 1, 1])
BUZZ3_test_X = BUZZ3_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ3_train_X.shape[0] == BUZZ3_train_Y.shape[0]
assert BUZZ3_test_X.shape[0]  == BUZZ3_test_Y.shape[0]
assert BUZZ3_valid_X.shape[0] == BUZZ3_valid_Y.shape[0]

def make_audio_convnet_model():
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_audio_convnet_model(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model
def __200X10X3_1conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=3,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 200,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 10,
                            activation='softmax',
                            name='fc_layer_2')
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_200X10_relu_1conv(learn_rate=0.04):
    return __200X10X3_1conv_template(learn_rate)
def load_200X10_relu_1conv(model_path, learn_rate=0.04):
    model = __200X10X3_1conv_template(learn_rate)
    model.load(model_path)
    return model
def __256X10X3_1conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=5,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 256,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 10,
                            activation='softmax',
                            name='fc_layer_2')
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_256X10_relu_1conv(learn_rate=0.04):
    return __256X10X3_1conv_template(learn_rate)
def load_256X10_relu_1conv(model_path, learn_rate=0.04):
    model = __256X10X3_1conv_template(learn_rate)
    model.load(model_path)
    return model
def __256X10X3_1conv_bigFilt_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=10,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 256,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 10,
                            activation='softmax',
                            name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_256X10_relu_1conv_bigFilt(learn_rate=0.04):
    return __256X10X3_1conv_bigFilt_template(learn_rate)
def load_256X10_relu_1conv_bigFilt(model_path, learn_rate=0.04):
    model = __256X10X3_1conv_bigFilt_template(learn_rate)
    model.load(model_path)
    return model
def __100X20X3_1conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=5,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 100,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 20,
                            activation='softmax',
                            name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_100X20X3_1conv(learn_rate=0.04):
    return __100X20X3_1conv_template(learn_rate)
def load_100X20X3_relu_1conv(model_path, learn_rate=0.04):
    model = __100X20X3_1conv_template(learn_rate)
    model.load(model_path)
    return model
def __100X20X3_1conv_bigfilt_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=10,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 100,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 20,
                            activation='softmax',
                            name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_100X20X3_1conv_bigfilt(learn_rate=0.04):
    return __100X20X3_1conv_bigfilt_template(learn_rate)
def load_100X20X2_relu_1conv_bigfilt(model_path, learn_rate=0.04):
    model = __100X20X3_1conv_bigfilt_template(learn_rate)
    model.load(model_path)
    return model
def __64X64X3_1conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=5,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 4, name='pool_layer')
    net = fully_connected(net, 64,
                            activation='relu',
                            name='fc_layer_1')
    net = fully_connected(net, 64,
                            activation='softmax',
                            name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_64X64X3_relu_1conv(learn_rate=0.04):
    return __64X64X3_1conv_template(learn_rate)
def load_64X64X3_relu_1conv(model_path, learn_rate=0.04):
    model = __64X64X3_1conv_template(learn_rate)
    model.load(model_path)
    return model
def __32X_1conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=10,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 2, name='pool_layer')
    net = fully_connected(net, 32,
                            activation='relu',
                            name='fc_layer_1')
    # net = fully_connected(net, ,
    #                         activation='softmax',
    #                         name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_32_relu_1conv(learn_rate=0.04):
    return __32X_1conv_template(learn_rate)
def load_32_relu_1conv(model_path, learn_rate=0.04):
    model = __32X_1conv_template(learn_rate)
    model.load(model_path)
    return model
def __100X3_2conv_template(learn_rate):
    net = input_data(shape=[None, 4000, 1, 1])
    net = conv_2d(net, nb_filter=10,
                        filter_size=5,
                        activation='relu',
                        name='conv_layer_1')
    net = max_pool_2d(net, 2, name='pool_layer')
    net = conv_2d(net, nb_filter=10,
                        filter_size=5,
                        activation='relu',
                        name='conv_layer_2')
    net = max_pool_2d(net, 2)
    net = fully_connected(net, 100,
                            activation='relu',
                            name='fc_layer_1')
    # net = fully_connected(net, ,
    #                         activation='softmax',
    #                         name='fc_layer_2')
    # net = dropout(net, 0.9)
    net = fully_connected(net, 3,
                            activation='softmax',
                            name='fc_layer_3')
    net = regression(net, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=learn_rate) 
    return tflearn.DNN(net)
def make_100X3_2conv(learn_rate=0.1):
    return __100X3_2conv_template(learn_rate)
def load_100X3_2conv(model_path, learn_rate=0.1):
    model = __100X3_2conv_template(learn_rate)
    model.load(model_path)
    return model
def test_tfl_audio_convnet_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 4000, 1, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))

###  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_audio_convnet_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.compat.v1.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='audio_cn_model')

### validating is testing on valid_X and valid_Y.
def validate_tfl_audio_convnet_model(model, valid_X, valid_Y):
    return test_tfl_audio_convnet_model(model, valid_X, valid_Y)
