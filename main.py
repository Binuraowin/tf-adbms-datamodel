%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bookstore-81666.appspot.com/o/bucket%2Fdata1.csv?alt=media&token=d4a0d2f6-44d0-4a86-8212-ade80bac92fb') # training data
dfeval = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bookstore-81666.appspot.com/o/bucket%2Fdata2.csv?alt=media&token=947a0a1c-60f2-4e26-9751-1a8e609c4905') # testing
print(dftrain.head())

y_train = dftrain.pop('paid')
y_eval = dfeval.pop('paid')

dftrain.PaymentAmount.hist(bins=10)

dftrain.ICode.value_counts().plot(kind='barh')
dftrain['RepID'].value_counts().plot(kind='barh')

pd.concat([dftrain, y_train], axis=1).groupby('RepID').PaymentAmount.mean().plot(kind='barh').set_xlabel('% PaymentAmount')

dftrain['ICode'].value_counts().plot(kind='barh')

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)