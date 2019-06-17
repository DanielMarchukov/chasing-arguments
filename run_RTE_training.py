from preprocessor import Preprocessor
from datasets import CheckDownloadUnzipData as data
from lstm import LSTM

import tensorflow as tf
import numpy as np

import os
import time
import datetime
import datasets


class TextualEntailment:
    def __init__(self):
        # print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
        #       " Checking for data sets, downloading if needed...")
        # data.check_all_unzip()

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing preproc...")
        self.__preproc = Preprocessor(evidence_length=30, hypothesis_length=30, vector_size=128)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Setting up GloVe word map...")
        self.__preproc.setup_word_map(file=datasets.glove_vectors_840B_300d)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing LSTM...")
        self.__lstm = LSTM(e_length=self.__preproc.get_evidence_length(),
                           h_length=self.__preproc.get_hypothesis_length(),
                           v_size=self.__preproc.get_vector_size())
        self.__lstm.setup_accuracy_scope()
        self.__lstm.setup_loss_scope()
        self.__tfsession = tf.Sesstion()

    def __del__(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Session Ended.")
        self.__tfsession.close()

    def run_predict(self, s1, s2):
        evi_sentence = [self.__preproc.fit_to_size(np.vstack(self.__preproc.sentence2sequence(s1)[0]),
                                                   (self.__preproc.get_evidence_length(), self.__preproc.get_vector_size()))]

        hyp_sentence = [self.__preproc.fit_to_size(np.vstack(self.__preproc.sentence2sequence(s2)[0]),
                                                   (self.__preproc.get_hypothesis_length(), self.__preproc.get_vector_size()))]

        tfsession = self.__lstm.load_session(model_path)
        result = self.__lstm.run_textual_entailment(evi_sentence, hyp_sentence, tfsession)
        return result

    def load_session(self, path_to_model):
        self.__lstm.load_session(path=path_to_model)

    def run_training(self, save_as):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Updating data scores for training...")
        data_feature_list, correct_scores = self.__preproc.update_data_scores(file=datasets.snli_full_dataset_file)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting training...")
        self.__lstm.train(df_list=data_feature_list, c_scores=correct_scores, save=save_as)

    def run_validation(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Updating data scores for validation...")
        data_feature_list, correct_scores = self.__preproc.update_data_scores(file=datasets.snli_dev_file)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting validation...")
        self.__lstm.validate(df_list=data_feature_list, c_scores=correct_scores, sess=self.__tfsession)

    def run_test(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Updating data scores for testing...")
        data_feature_list, correct_scores = self.__preproc.update_data_scores(file=datasets.snli_test_file)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting testing...")
        self.__lstm.test(df_list=data_feature_list, c_scores=correct_scores, sess=self.__tfsession)


rte = TextualEntailment()
rte.run_training(save_as="rnn-128d-1024h-lr-001-final")

# rnn-128d-1024h-out-10-lr-001-final
model_path = os.getcwd() + '\\models\\rnn-840B-128d-1024h-final'
restore = False
if restore:
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Restoring model...")
    rte.load_session(model_path)

rte.run_validation()
rte.run_test()