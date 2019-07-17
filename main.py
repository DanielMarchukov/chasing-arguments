from gather_tweets import TwitterMining
from textual_entailment import TextualEntailment
from arg_framework import ArgFramework

import datetime
import time
import csv
import os


class Main:
    def __init__(self, use_sentences=False):
        self.__af = ArgFramework()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Setting up Twitter...")
        self.__twitter = TwitterMining()
        self.__rte = None
        self.__lstm_results_csv = os.getcwd() + "\\logs\\LSTM_PREDICTION_RESULTS.csv"
        self.__use_sentences = use_sentences
        self.__sentences = []

    def setup_twitter_data(self, tweet_count):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Mining data...")
        if self.__use_sentences:
            self.__twitter.mine_sentences(count=tweet_count)
        else:
            self.__twitter.mine_tweets(count=tweet_count)

    def train_lstm_model(self, model_name):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
              + " Textual Entailment Training setup...")
        self.__rte = TextualEntailment(is_training=True)
        self.__rte.train_valid_test_log(save_as=model_name)

    def load_lstm_model(self, model_name):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
              + " Textual Entailment Setup...")
        self.__rte = TextualEntailment(is_training=False)
        model_path = os.getcwd() + '\\models\\' + model_name
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Restoring model...")
        self.__rte.load_model(model_path)

    def load_twitter_data(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Reading sentences...")
        filepath = self.__twitter.sentences_csv_path if self.__use_sentences else self.__twitter.tweets_csv_path
        with open(filepath, mode='r', encoding="UTF-8") as file:
            for line in file:
                sent = line.split("|")[-1].rstrip()
                self.__sentences.append(sent)

    def run_model_on_twitter(self):
        relation_count = 0
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Doing predictions...")
        with open(self.__lstm_results_csv, mode="w", encoding="UTF-8", newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.__sentences) - 1, 0, -1):
                for j in range(i - 1, 0, -1):
                    res = self.__rte.run_predict(self.__sentences[j], self.__sentences[i])
                    if res == "E" or res == "C":
                        relation_count = relation_count + 1
                        self.__af.add_argument(self.__sentences[j])
                        self.__af.add_argument(self.__sentences[i])
                        self.__af.add_relation(self.__sentences[j], self.__sentences[i],
                                               'red' if res in ["C"] else 'green')
                        file.writerow([j, res, i, self.__sentences[j], self.__sentences[i]])
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Calculated " +
              str(relation_count) + " Argument Relations...")

    def print_subsets(self):
        self.__af.conflict_free_arguments()

    def save_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Saving...")
        self.__af.save()

    def draw_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Drawing...")
        self.__af.draw()


def main():
    application = Main(use_sentences=False)
    # application.setup_twitter_data(tweet_count=15)
    application.train_lstm_model(model_name="RNN_vs128_b256_hs768_ml30")
    # application.load_lstm_model(model_name="RNN_vs128_b256_hs1024_ml30")
    # application.load_twitter_data()
    # application.run_model_on_twitter()

    # application.print_subsets()

    # application.save_argument_framework()
    # application.draw_argument_framework()


if __name__ == "__main__":
    main()
