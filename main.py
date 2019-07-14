from gather_tweets import TwitterMining
from textual_entailment import TextualEntailment
from arg_framework import ArgFramework

import datetime
import time
import csv
import os


class Main:
    def __init__(self):
        self.__af = ArgFramework()
        self.__twitter = None
        self.__rte = None
        self.__sentences = []

    def setup_twitter_data(self, tweet_count):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Connecting to Twitter...")
        self.__twitter = TwitterMining()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Mining data...")
        self.__twitter.mine_data(count=tweet_count)

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

    def load_twitter_sentences(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Reading sentences...")
        with open(self.__twitter.tweets_csv_path, mode='r', encoding="UTF-8") as file:
            for line in file:
                sent = line.split("|")[-1].rstrip()
                # sent = split[-1].rstrip()
                self.__sentences.append(sent)

    def run_model_on_twitter(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Doing predictions...")
        with open("lstm_results.csv", mode="w", encoding="UTF-8", newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.__sentences)):
                for j in range(i + 1, len(self.__sentences)):
                    res = self.__rte.run_predict(self.__sentences[j], self.__sentences[i])
                    print(res)
                    if res == "E" or res == "C":
                        self.__af.add_argument(self.__sentences[j])
                        self.__af.add_argument(self.__sentences[i])
                        self.__af.add_relation(self.__sentences[j], self.__sentences[i], 'r' if res in ["C"] else 'g')
                        file.writerow([j, self.__sentences[j], res, i, self.__sentences[i]])

    def save_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Saving...")
        self.__af.save()

    def draw_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Drawing...")
        self.__af.draw()


def main():
    application = Main()
    # application.setup_twitter_data(tweet_count=100)
    application.train_lstm_model(model_name="RNN_vs216_b256_hs1024_ml30")
    # application.load_lstm_model(model_name="RNN_vs216_b256_hs1024_ml30")
    # application.load_twitter_sentences()
    # application.run_model_on_twitter()
    # application.save_argument_framework()
    # application.draw_argument_framework()
    # do analysis


if __name__ == "__main__":
    main()
