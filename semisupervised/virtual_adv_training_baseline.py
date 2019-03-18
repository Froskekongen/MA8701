# coding: utf8
"""
Note (Erlend Aune): This example was taken from
https://gist.github.com/mokemokechicken/2658d036c717a3ed07064aa79a59c82d
It needs to be modified to run in semi-supervised mode.


* VAT: https://arxiv.org/abs/1507.00677

# 参考にしたCode
Original: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
VAT: https://github.com/musyoku/vat/blob/master/vat.py

results example
---------------

finish: use_dropout=False, use_vat=False: score=0.215942835068, accuracy=0.9872
finish: use_dropout=True, use_vat=False: score=0.261140023788, accuracy=0.9845
finish: use_dropout=False, use_vat=True: score=0.240192672965, accuracy=0.9894
finish: use_dropout=True, use_vat=True: score=0.210011005498, accuracy=0.9891
"""
import numpy as np
from functools import reduce
from keras.utils.generic_utils import to_list
from keras.engine.training import Model

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

SAMPLE_SIZE = 0

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def main(data, use_dropout, use_vat):
    np.random.seed(1337)  # for reproducibility

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = data

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if SAMPLE_SIZE:
        X_train = X_train[:SAMPLE_SIZE]
        y_train = y_train[:SAMPLE_SIZE]
        X_test = X_test[:SAMPLE_SIZE]
        y_test = y_test[:SAMPLE_SIZE]

    print("start: use_dropout=%s, use_vat=%s" % (use_dropout, use_vat))
    my_model = MyModel(input_shape, use_dropout, use_vat).build()
    my_model.training(X_train, y_train, X_test, y_test)

    score = my_model.model.evaluate(X_test, y_test, verbose=0)
    print("finish: use_dropout=%s, use_vat=%s: score=%s, accuracy=%s" % (use_dropout, use_vat, score[0], score[1]))


class MyModel:
    model = None

    def __init__(self, input_shape, use_dropout=True, use_vat=True):
        self.input_shape = input_shape
        self.use_dropout = use_dropout
        self.use_vat = use_vat

    def build(self):
        input_layer = Input(self.input_shape)
        output_layer = self.core_data_flow(input_layer)
        if self.use_vat:
            self.model = VATModel(input_layer, output_layer).setup_vat_loss()
        else:
            self.model = Model(input_layer, output_layer)
        return self

    def core_data_flow(self, input_layer):
        x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid')(input_layer)
        x = Activation('relu')(x)
        x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        if self.use_dropout:
            x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        if self.use_dropout:
            x = Dropout(0.5)(x)
        x = Dense(nb_classes, activation='softmax')(x)
        return x

    def training(self, X_train, y_train, X_test, y_test):
        self.model.compile(loss=K.categorical_crossentropy, optimizer='adadelta', metrics=['accuracy'])
        np.random.seed(1337)  # for reproducibility
        self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                       verbose=1, validation_data=(X_test, y_test))


class VATModel(Model):
    _vat_loss = None

    def setup_vat_loss(self, eps=1, xi=10, ip=1):
        self._vat_loss = self.vat_loss(eps, xi, ip)
        return self

    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self._vat_loss:
            losses += [self._vat_loss]
        return losses

    def vat_loss(self, eps, xi, ip):
        normal_outputs = [K.stop_gradient(x) for x in to_list(self.outputs)]
        d_list = [K.random_normal(x.shape) for x in self.inputs]

        for _ in range(ip):
            new_inputs = [x + self.normalize_vector(d)*xi for (x, d) in zip(self.inputs, d_list)]
            new_outputs = to_list(self.call(new_inputs))
            klds = [K.sum(self.kld(normal, new)) for normal, new in zip(normal_outputs, new_outputs)]
            kld = reduce(lambda t, x: t+x, klds, 0)
            d_list = [K.stop_gradient(d) for d in K.gradients(kld, d_list)]

        new_inputs = [x + self.normalize_vector(d) * eps for (x, d) in zip(self.inputs, d_list)]
        y_perturbations = to_list(self.call(new_inputs))
        klds = [K.mean(self.kld(normal, new)) for normal, new in zip(normal_outputs, y_perturbations)]
        kld = reduce(lambda t, x: t + x, klds, 0)
        return kld

    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, dim=-1)
        return x / (K.sqrt(z) + K.epsilon())

    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)


data = mnist.load_data()
main(data, use_dropout=False, use_vat=False)
main(data, use_dropout=True, use_vat=False)
main(data, use_dropout=False, use_vat=True)
main(data, use_dropout=True, use_vat=True)
