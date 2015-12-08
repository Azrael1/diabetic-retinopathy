import os
import sys
import timeit
import re

import numpy as np
import csv

import cv2
import time

import matplotlib
import matplotlib.pyplot as pyplot
from pylab import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def load_data(image_dataset, label_dataset, image_size):
    """
    Loads the dataset. The images don't load in sequence so this function is used to correct that part.
    Does the preprocessing to align the image data with labels data and then returns them
    type dataset: str
    param dataset: symbolic link to data
    """
    # For now, load the dataset from the train folder.
    numbers = re.compile(r'(\d+)')
    label_reader = csv.reader(open(label_dataset, mode='rb'))
    # Discard the 1st field in -- ['image', 'level']
    labels = [label for label in label_reader][1:]

    def numericalsort(value):
        x = numbers.split(value)
        # example of a split -- ['', '10003', '_left.tiff']
        x[1::3] = map(int, x[1::3])
        return x

    data_x = []
    images = os.listdir(image_dataset)
    sorted_images = sorted(images, key=numericalsort)
    idx = 0
    for image in sorted_images:
        image_link = image_dataset + '/' + image
        opened_image = cv2.imread(image_link)
        data_x.append(opened_image)
        assert image.split('.')[0] == labels[idx][0]
        idx += 1
    data_x = np.asarray(data_x, theano.config.floatX)
#    data_x.shape = [num_images, num_channels, num_rows, num_columns] -- [35126, 3, image_size, image_size]
    assert data_x.shape == (35126, image_size, image_size, 3)
    y = [label[1] for label in labels]
    data_y = np.asarray(y, 'int32')

    assert data_y.shape[0] == 35126
    return data_x, data_y, len(y)


def load_balanced_dataset(image_dataset, label_dataset, image_size):
    """
    Loads balanced version of dataset. Meaning each label will be having equal amount of images in the data.
    type dataset: str
    param dataset: symbolic link to data
    """
    # For now, load the dataset from the train folder.
    # The constraint for a fully balanced dataset(Each class having equal amounts of data) is imposed by level 4
    # (0-4 levels). It possesses only 700 images(2 percent of the data).
    # So I am going to take the first 700 images from each label and only train on them.

    numbers = re.compile(r'(\d+)')
    label_reader = csv.reader(open(label_dataset, mode='rb'))
    # Discard the 1st field in -- ['image', 'level']
    labels = [label for label in label_reader][1:]

    def numericalsort(value):
        x = numbers.split(value)
        # example of a split -- ['', '10003', '_left.tiff']
        x[1::3] = map(int, x[1::3])
        return x

    data_x, y = [], []
    images = os.listdir(image_dataset)
    sorted_images = sorted(images, key=numericalsort)
    idx = 0
    # idx_ is the variable that keeps count of the number of images that have been seen per category.
    idx_ = [0]*5
    for image in sorted_images:
        assert image.split('.')[0] == labels[idx][0]
        if idx_[int(labels[idx][1])] < 700:
            idx_[int(labels[idx][1])] += 1
            y.append(int(labels[idx][1]))
            image_link = image_dataset + '/' + image
            opened_image = cv2.imread(image_link)
            data_x.append(opened_image)

        idx += 1
    data_x = np.asarray(data_x, theano.config.floatX)

    # 3500 because there are 5 labels and each labels has 700 images.
    assert data_x.shape == (3500, image_size, image_size, 3)
    data_y = np.asarray(y, 'int32')

    assert data_y.shape[0] == 3500
    return data_x, data_y, len(y)


def evaluate_lenet5_grads(learning_rate=0.013, n_epochs=700,
                    dataset='/media/azrael/Data/data/train_res_smallest',
                    label_dataset='/home/azrael/Documents/ubuntu/trainLabels.csv',
                    nkerns=[20, 50, 50], batch_size=20):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = np.random.RandomState(23455)
    image_size = 32

    data_x, data_y, len_data = load_balanced_dataset(dataset, label_dataset, image_size)
    assert data_x.ndim == 4
    assert data_y.ndim == 1

    train_split, valid_split = int(len_data * 0.4), int(len_data * 0.8)
    print 'Train_split', train_split
    print 'Valid_split', valid_split

    train_set_x, train_set_y = data_x[:train_split, :, :, :], data_y[:train_split]
    valid_set_x, valid_set_y = data_x[train_split: valid_split, :, :, :], data_y[train_split: valid_split]
    test_set_x, test_set_y = data_x[valid_split:, :, :, :], data_y[valid_split:]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    print 'N_test_batches', n_test_batches
    print 'N_train_batches', n_train_batches
    print 'N_valid_batches', n_valid_batches

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    train_set_x = theano.shared(train_set_x, name='train_set_x', borrow=True).dimshuffle(0, 3, 1, 2)
    train_set_y = theano.shared(train_set_y, name='train_set_x', borrow=True)
    valid_set_x = theano.shared(valid_set_x, name='valid_set_x', borrow=True).dimshuffle(0, 3, 1, 2)
    valid_set_y = theano.shared(valid_set_y, name='valid_set_y', borrow=True)
    test_set_x = theano.shared(test_set_x, name='test_set_x', borrow=True).dimshuffle(0, 3, 1, 2)
    test_set_y = theano.shared(test_set_y, name='test_set_y', borrow=True)

    # start-snippet-1

    x = T.tensor4('x', theano.config.floatX)   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layer0_input = x

    # For the Conv Layer:
    # Accepts a volume of size (w1, h1, d1)
    # Requires 4 hyperparameters
    # -- Number of filters K
    # -- Their Spatial extent F
    # -- The stride S
    # -- Amount of zero padding P
    # Produces a volume of size (w2, h2, d2)
    # w2 = (w1- F + 2P)/S + 1
    # h2 = (h1- F + 2P)/S + 1
    # d2 = K

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0],)
    poolfactor_layer0 = 2

    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, image_size, image_size),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(poolfactor_layer0, poolfactor_layer0)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 13 * 13),
    # or (500, 50 * 13 * 13) with the default values.
    # theano's flatten has different behaviour compared to np's.
    layer0__input = layer0.output.flatten(2)

    layer0_ = HiddenLayer(
        rng,
        input=layer0__input,
        n_in=nkerns[0] * 15 * 15,
        n_out=500,
        activation=T.tanh
    )

    # construct a fully-connected sigmoidal layer
    # layer0 directly connected with layer3. Nothing to get confused about, I just insert layers between 0 and 3
    # when I need to and hence prevent unneccessary renumbering.
    layer3 = HiddenLayer(
        rng,
        input=layer0_.output,
        n_in=500,
        n_out=500,
        activation=T.tanh
    )
    # shape of layer3.output = (batch_size,500)

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(rng, input=layer3.output, n_in=500, n_out=5)
    # shape of layer4.output = (batch_size, 5)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        [layer0.output, cost, layer4.errors(y)],
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params  + layer0_.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        [y, layer4.errors(y), cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    validation_frequency = n_epochs
    best_validation_error = np.inf
    best_test_score = np.inf

    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    iter = 0
    t_loss, t_error, v_loss, v_error = [], [], [], []
    while (epoch < n_epochs):
        epoch = epoch + 1
        training_loss = []
        training_errors = []
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            pred = train_model(minibatch_index)
            training_loss.append(pred[1])
            training_errors.append(pred[2])
        this_training_loss = np.mean(np.array(training_loss))
        this_training_error = np.mean(np.array(training_errors))
        print('epoch %i,  training loss %f, training error %f %%' %
              (epoch, this_training_loss, this_training_error*100))
        t_loss.append(this_training_loss)
        t_error.append(this_training_error)

        # Each image of the minibatch is stored in a single index in validation_images.
        validation_images = [validate_model(i)[0] for i
                             in xrange(n_valid_batches)]
        validation_losses = [validate_model(i)[1] for i
                             in xrange(n_valid_batches)]
        validation_errors = [validate_model(i)[2] for i
                             in xrange(n_valid_batches)]

        this_validation_loss = np.mean(validation_losses)
        this_validation_error = np.mean(validation_errors)
        print('epoch %i,  validation loss %f, validation error %f %%' %
              (epoch, this_validation_loss, this_validation_error*100))

        v_loss.append(this_validation_loss)
        v_error.append(this_validation_error)
        if this_validation_error < best_validation_error:
            best_validation_error = this_validation_error
            images_level0 = np.array(validation_images)

            test_errors = [
                    test_model(i)
                    for i in xrange(n_test_batches)
            ]
            test_score = np.mean(test_errors)
            print test_score
            if (test_score < best_test_score):
                best_test_score = test_score
                idox = 0
                for image in images_level0:
                    x = np.mean(np.mean(image, axis=0), axis=1)
                    x_min = x.min()
                    x_max = x.max()
                    current_value = x-x_min
                    newmin = 0
                    scale=(x-x_min)/(x_max-x_min)

                    newvalue=(255*scale)+newmin
                    cv2.imwrite('/media/azrael/Data/data/output_level0/'+str(idox)+'.png', newvalue)
                    idox +=1

    print('test error of best model',
          best_test_score * 100.)
    plt.figure(1)
    plt.plot(t_error)
    plt.plot(t_loss)
    plt.legend(['training error', 'training loss'])
    plt.figure(2)
    plt.plot(v_error)
    plt.plot(v_loss)
    plt.legend(['validation loss', 'validation error'])
    pyplot.show()

if __name__ == '__main__':
    evaluate_lenet5_grads()
