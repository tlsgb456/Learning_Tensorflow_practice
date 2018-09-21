import tensorflow as tf
from tensorflow.contrib import learn

# model = learn.DNNRegressor()
# model.fit()
# model.evaluate()
# model.predict()

# # -- 7.2.1 --
# from sklearn import datasets, metrics, preprocessing
# boston = datasets.load_boston()
# x_data = preprocessing.StandardScaler().fit_transform(boston.data)
# y_data = boston.target
#
# NUM_STEPS = 200
# MINIBATCH_SIZE = 506
#
# # some represent of data is instanced which named 'feature_columns'.
# feature_columns = learn.infer_real_valued_columns_from_input(x_data)
#
# # use estimator
# reg = learn.LinearRegressor(
#     feature_columns=feature_columns,
#     optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
#     )
# reg.fit(x_data, boston.target, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)
#
# # return MSE loss value
# MSE = reg.evaluate(x_data, boston.target, steps=1)
#
# print(MSE)

# -- 7.3.2 --
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

with tf.device('/cpu:0'):
    X, Y, X_test, Y_test = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    X_test = X_test.reshape([-1, 28, 28, 1])

    # CNN network
    CNN = input_data(shape=[None, 28, 28, 1], name='input')
    CNN = conv_2d(CNN, 32, 5, activation='LeakyReLU', regularizer="L2")
    CNN = max_pool_2d(CNN, 2)
    CNN = batch_normalization(CNN)
    CNN = conv_2d(CNN, 64, 5, activation='LeakyReLU', regularizer="L2")
    CNN = max_pool_2d(CNN, 2)
    CNN = batch_normalization(CNN)
    CNN = fully_connected(CNN, 1024, activation=None)
    CNN = batch_normalization(CNN)
    CNN = fully_connected(CNN, 10, activation='softmax')
    CNN = regression(CNN, optimizer='adam', learning_rate=1e-4, loss='categorical_crossentropy', name='target')

    # CNN network training
    model = tflearn.DNN(CNN, tensorboard_verbose=0,
                        tensorboard_dir='./MNIST_tflearn_board/',
                        checkpoint_path='./MNIST_tflearn_checkpoints/checkpoint/')
    model.fit({'input': X}, {'target': Y}, n_epoch=3,
              validation_set=({'input': X_test}, {'target': Y_test}),
              snapshot_step=1000, show_metric=True, run_id='convnet_mnist')




