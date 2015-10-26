
# work on green channel only.
import os
import sys
import timeit
import re

import numpy as np
import csv

import cv2

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


def load_data(image_dataset, label_dataset):
    """
    Loads the dataset.
    type dataset: str
    param dataset: symbolic link to data
    """
    # For now, load the dataset from the train folder.
    numbers = re.compile(r'(\d+)')
    label_reader = csv.reader(open(label_dataset, mode='rb'))
    # Discard the 1st field which is ['image', 'level']
    labels = [label for label in label_reader][1:]

    def numericalsort(value):
        x = numbers.split(value)
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
#    data_x.shape = [num_images, num_channels, num_rows, num_columns] -- [35126, 3, 64, 64]
    assert data_x.shape == (35126, 64, 64, 3)
    y = [label[1] for label in labels]
    data_y = np.asarray(y, 'int32')
    assert data_y.shape[0] == 35126
    return data_x, data_y, len(y)


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='/media/azrael/Data/data/train_res_smaller',
                    label_dataset='/home/azrael/Documents/ubuntu/trainLabels.csv',
                    nkerns=[20, 50, 50], batch_size=500):
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

    data_x, data_y, len_data = load_data(dataset, label_dataset)
    assert data_x.ndim == 4
    assert data_y.ndim == 1

    train_split, valid_split = int(len_data * 0.6), int(len_data * 0.9)

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
    image_size = 64

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
    # filtering reduces the image size to (64-5+1 , 64-5+1) = (60, 60)
    # maxpooling reduces this further to (60/2, 60/2) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 30, 30)
    poolfactor_layer0 = 2

    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, image_size, image_size),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(poolfactor_layer0, poolfactor_layer0)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (30-5+1, 30-5+1) = (26, 26)
    # maxpooling reduces this further to (26/2, 26/2) = (13, 13)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 13, 13)
    poolfactor_layer1 = 2

    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 30, 30),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(poolfactor_layer1, poolfactor_layer1)
    )

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (61-2+1, 61-2+1) = (60, 60)
    # maxpooling reduces this further to (60/4, 60/2) = (15,15)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 15, 15)
    # layer2 = ConvPoolLayer(
    #    rng,
    #    input=layer0.output,
    #    image_shape=(batch_size, nkerns[1], 61, 61),
    #    filter_shape=(nkerns[2], nkerns[1], 15, 15),
    #    poolsize=(4, 4)
    #)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 13 * 13),
    # or (500, 50 * 13 * 13) = (500, ??) with the default values.
    # theano's flatten has different behaviour compared to np's.
    layer3_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[1] * 13 * 13,
        n_out=500,
        activation=T.tanh
    )
    # shape of layer3.output = (batch_size,500)

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=5)
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
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer1.params + layer0.params

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
        cost,
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
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
