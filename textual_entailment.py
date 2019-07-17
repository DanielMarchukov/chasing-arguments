from preprocessor import Preprocessor
from datasets import CheckDownloadUnzipData as data
from lstm import LSTM

import numpy as np

import time
import datetime
import datasets

VECTOR_SIZE = 128
EVIDENCE_LENGTH = 30
HYPOTHESIS_LENGTH = 30


class TextualEntailment:
    def __init__(self, is_training=False):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Checking for data sets, downloading if needed...")
        data.check_all_unzip()

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing preproc...")
        self.__preproc = Preprocessor(evidence_length=EVIDENCE_LENGTH, hypothesis_length=HYPOTHESIS_LENGTH,
                                      vector_size=VECTOR_SIZE)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Setting up GloVe word map...")
        self.__preproc.setup_word_map(file=datasets.glove_vectors_840B_300d)
        self.__df_list = None
        self.__c_scores = None

        if is_training:
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
                  " Updating data scores for training...")
            self.__df_list, self.__c_scores = self.__preproc.update_data_scores(file=datasets.snli_full_dataset_file)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Initializing LSTM...")
        self.__lstm = LSTM(e_length=self.__preproc.get_evidence_length(),
                           h_length=self.__preproc.get_hypothesis_length(),
                           v_size=self.__preproc.get_vector_size())

    def run_predict(self, s1, s2):
        evi_sentence = [self.__preproc.fit_to_size(np.vstack(self.__preproc.sentence2sequence(s1)[0]),
                                                   (self.__preproc.get_evidence_length(),
                                                    self.__preproc.get_vector_size()))]

        hyp_sentence = [self.__preproc.fit_to_size(np.vstack(self.__preproc.sentence2sequence(s2)[0]),
                                                   (self.__preproc.get_hypothesis_length(),
                                                    self.__preproc.get_vector_size()))]

        result = self.__lstm.run_prediction(evi_sentence, hyp_sentence)
        return result

    def load_model(self, path_to_model):
        self.__lstm.load_model(path=path_to_model)

    def run_training(self, save_as):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting training...")
        self.__lstm.train(df_list=self.__df_list, c_scores=self.__c_scores, save=save_as)

    def run_validation(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Updating data scores for validation...")
        self.__df_list, self.__c_scores = self.__preproc.update_data_scores(file=datasets.snli_dev_file)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting validation...")
        self.__lstm.validate(df_list=self.__df_list, c_scores=self.__c_scores)

    def run_test(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
              " Updating data scores for testing...")
        self.__df_list, self.__c_scores = self.__preproc.update_data_scores(file=datasets.snli_test_file)

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Starting testing...")
        self.__lstm.test(df_list=self.__df_list, c_scores=self.__c_scores)

    def train_valid_test_log(self, save_as):
        self.run_training(save_as=save_as)
        self.run_validation()
        self.run_test()
        self.__lstm.log_results()
