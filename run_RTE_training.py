from datasets import CheckDownloadUnzipData
from model import Model
from lstm import LSTM

import tensorflow as tf

import os
import time
import datetime

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
      " Checking for data sets, downloading if needed...")
# CheckDownloadUnzipData.check_all_unzip()

training = True
restore = False
model_path = os.getcwd() + '\\models\\rnn-840B-300d-6m-300-final'

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing model...")
model = Model()

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Setting up GloVe word map...")
model.setup_word_map(file="\\data\\GloVe\\glove.840B.300d.txt")

if training:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
          " Splitting data into scores for training...")
    data_feature_list, correct_scores = model.split_data_into_scores(file=snli_full_dataset_file)

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing LSTM...")
lstm = LSTM(model=model)
lstm.setup_accuracy_scope()
lstm.setup_loss_scope()
sess = tf.Session()

if training:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting training...")
    sess = lstm.train(data_feature_list=data_feature_list,
                      correct_scores=correct_scores,
                      save="rnn-840B-300d-6m-300-final")

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
      " Splitting data into scores for validation...")
data_feature_list, correct_scores = model.split_data_into_scores(file=snli_dev_file)
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
data_feature_list, correct_scores = model.split_data_into_scores(file=snli_test_file)
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting testing...")
lstm.test(data_feature_list=data_feature_list,
          correct_scores=correct_scores,
          sess=sess)

# evidences = ["John was injured in a car crash.", "Peter is a tall person.", "Jack is loved by everyone."]
# hypotheses = ["John had an accident.", "Josh plays football.", "Everybody hates Jack."]
#
# for i in range(len(evidences)):
#     sentence1 = [rte.fit_to_size(np.vstack(rte.sentence2sequence(evidences[i], glove_wordmap)[0]),
#                                  (max_evidence_length, vector_size))]
#     sentence2 = [rte.fit_to_size(np.vstack(rte.sentence2sequence(hypotheses[i], glove_wordmap)[0]),
#                                  (max_hypothesis_length, vector_size))]
#     prediction = sess.run(classification_scores, feed_dict={hyp: (sentence1 * batch_size),
#                                                             evi: (sentence2 * batch_size),
#                                                             y: [[0, 0, 0]] * batch_size})
#
#     print(["Entails", "Null", "Contradicts"][np.argmax(prediction[0])])

sess.close()
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Session Ended.")