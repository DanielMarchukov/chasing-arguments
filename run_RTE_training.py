from six.moves.urllib.request import urlretrieve
from model import Model
from lstm import LSTM

import tensorflow as tf

import os
import time
import datetime
import zipfile

glove_zip_file = "\\data\\GloVe\\glove.6B.zip"
glove_vectors_50d = "\\data\\GloVe\\glove.6B.50d.txt"
glove_vectors_100d = "\\data\\GloVe\\glove.6B.100d.txt"
glove_vectors_200d = "\\data\\GloVe\\glove.6B.200d.txt"
glove_vectors_300d = "\\data\\GloVe\\glove.6B.300d.txt"
glove_vectors_840B_300d = "\\data\\GloVe\\glove.840B.300d.txt"
snli_zip_file = "\\data\\SNLI\\snli_1.0.zip"

snli_dev_file = "\\data\\SNLI\\snli_1.0_dev.txt"
snli_test_file = "\\data\\SNLI\\snli_1.0_test.txt"
snli_full_dataset_file = "\\data\\SNLI\\snli_1.0_train.txt"

print("Checking for data sets, downloading if needed...")


# def unzip_single_file(zip_file_name, output_file_name):
#     if not os.path.isfile(output_file_name):
#         with open(output_file_name, 'wb') as out_file:
#             with zipfile.ZipFile(zip_file_name) as zipped:
#                 for info in zipped.infolist():
#                     if output_file_name in info.filename:
#                         with zipped.open(info) as requested_file:
#                             out_file.write(requested_file.read())
#                             return
#
#
# if not os.path.isfile(glove_zip_file) and (not os.path.isfile(glove_vectors_50d)
#                                            or not os.path.isfile(glove_vectors_100d)
#                                            or not os.path.isfile(glove_vectors_200d)
#                                            or not os.path.isfile(glove_vectors_300d)):
#     urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", glove_zip_file)
#
# if not os.path.isfile(snli_zip_file) and (not os.path.isfile(snli_full_dataset_file)
#                                           or not os.path.isfile(snli_dev_file)
#                                           or not os.path.isfile(snli_test_file)):
#     urlretrieve("https://nlp.stanford.edu/projects/snli/snli_1.0.zip", snli_zip_file)
#
# unzip_single_file(glove_zip_file, glove_vectors_50d)
# unzip_single_file(glove_zip_file, glove_vectors_100d)
# unzip_single_file(glove_zip_file, glove_vectors_200d)
# unzip_single_file(glove_zip_file, glove_vectors_300d)
# unzip_single_file(glove_zip_file, glove_vectors_840B_300d)
# unzip_single_file(snli_zip_file, snli_full_dataset_file)
# unzip_single_file(snli_zip_file, snli_dev_file)
# unzip_single_file(snli_zip_file, snli_test_file)

training = True
restore = False
model_path = os.getcwd() + '\\models\\rnn-840B-300d-6m-300-final'

print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing model...")
model = Model()
model.vector_size = 200

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