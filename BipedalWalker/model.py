import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import config


def get_policy_model(input_size, output_size):
    model = Sequential()
    model.add(Input(shape = (config.FEED_MEM, input_size)))
    model.add(Dense(64, activation = 'elu'))
    model.add(Dense(32, activation = 'elu'))
    model.add(Dense(32, activation = 'elu'))
    model.add(Flatten())
    model.add(Dense(16, activation = 'elu'))
    model.add(Dense(8, activation = 'elu'))
    model.add(Dense(output_size*2, activation = 'linear'))
    model.compile(loss = tf.keras.losses.Huber(), optimizer = Adam())
    return model


def get_value_model(input_size):
    model = Sequential()
    regularizer = tf.keras.regularizers.L2(config.L2_NORM)
    model.add(Input(shape = (config.FEED_MEM, input_size)))
    model.add(Flatten())
    model.add(Dense(64, activation ='elu', kernel_regularizer = regularizer))
    model.add(Dense(32, activation ='elu', kernel_regularizer = regularizer))
    model.add(Dense(32, activation ='elu', kernel_regularizer = regularizer))
    model.add(Flatten())
    model.add(Dense(16, activation = 'elu', kernel_regularizer = regularizer))
    model.add(Dense(8, activation = 'elu', kernel_regularizer = regularizer))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = tf.keras.losses.Huber(), optimizer = Adam())
    return model
