from params import Params
from lstm import LSTM

import tensorflow as tf

import os
import time
import datetime
import datasets

# print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
#       " Checking for data sets, downloading if needed...")
# data.check_all_unzip()

training = True
restore = False
model_path = os.getcwd() + '\\models\\rnn-840B-128d-1024h-final'

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing model...")
model = Params()

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Setting up GloVe word map...")
model.setup_word_map(file=datasets.glove_vectors_840B_300d)

if training:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
          " Splitting data into scores for training...")
    data_feature_list, correct_scores = model.update_data_scores(file=datasets.snli_full_dataset_file)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing LSTM...")
lstm = LSTM(params=model)
lstm.setup_accuracy_scope()
lstm.setup_loss_scope()
sess = tf.Session()

if training:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting training...")
    sess = lstm.train(data_feature_list=data_feature_list,
                      correct_scores=correct_scores,
                      save="rnn-128d-1024h-out-10-final")

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
      " Splitting data into scores for validation...")
data_feature_list, correct_scores = model.update_data_scores(file=datasets.snli_dev_file)

if restore:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
          " Restoring model for validation...")
    tf.train.Saver().restore(sess, model_path)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting validation...")
lstm.validate(data_feature_list=data_feature_list,
              correct_scores=correct_scores,
              sess=sess)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
      " Splitting data into scores for testing...")
data_feature_list, correct_scores = model.update_data_scores(file=datasets.snli_test_file)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting testing...")
lstm.test(data_feature_list=data_feature_list,
          correct_scores=correct_scores,
          sess=sess)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Session Ended.")
sess.close()
