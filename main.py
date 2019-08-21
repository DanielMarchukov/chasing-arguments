from gather_tweets import TwitterMining
from textual_entailment import TextualEntailment
from arg_framework import ArgFramework, ATTACK, SUPPORT

import datetime
import warnings
import time
import csv
import os

warnings.filterwarnings("ignore")


class Main:
    def __init__(self):
        self.__af = ArgFramework()
        self.__twitter = None
        self.__rte = None
        self.__lstm_results_csv = os.getcwd() + "\\logs\\LSTM_PREDICTION_RESULTS.csv"
        self.__sentences = []
        self.__result = None

    def setup_twitter_data(self, tweet_count, q):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Setting up Twitter...")
        self.__twitter = TwitterMining()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Mining data...")
        self.__twitter.mine_tweets(query=q, since=str(datetime.datetime.now()).split(" ")[0], count=tweet_count)

    def train_lstm_model(self, model_name):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Textual Entailment "
                                                                                           "Training setup...")
        self.__rte = TextualEntailment(is_training=True)
        self.__rte.train_valid_test_log(save_as=model_name)

    def load_lstm_model(self, model_name):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Textual Entailment "
                                                                                           "Setup...")
        self.__rte = TextualEntailment(is_training=False)
        model_path = os.getcwd() + '\\models\\' + model_name
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Loading model...")
        self.__rte.load_model(model_path)

    def load_twitter_data(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Reading sentences...")
        with open(self.__twitter.tweets_csv_path, mode='r', encoding="UTF-8") as file:
            for line in file:
                sent = line.split("|")[-1].rstrip()
                self.__sentences.append(sent)

    def run_model_on_twitter(self):
        relation_count = 0
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Doing predictions...")
        with open(self.__lstm_results_csv, mode="w", encoding="UTF-8", newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.__sentences) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    res = self.__rte.run_predict(self.__sentences[j], self.__sentences[i])
                    if res == "E" or res == "C":
                        relation_count = relation_count + 1
                        self.__af.add_argument(self.__sentences[j])
                        self.__af.add_argument(self.__sentences[i])
                        self.__af.add_relation(self.__sentences[j], self.__sentences[i],
                                               ATTACK if res in ["C"] else SUPPORT)
                        file.writerow([j, res, i, self.__sentences[j], self.__sentences[i]])
        self.__af.pre_setup()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Created " +
              str(len(self.__af.get_af().nodes())) + " Argument Nodes...")
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Calculated " +
              str(relation_count) + " Argument Relations...")

    def print_subsets(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Analysing Conflict Free "
                                                                                           "Arguments...")
        self.__af.conflict_free_arguments()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Analysing Admissible "
                                                                                           "Arguments...")
        self.__af.admissible_arguments()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Computing Complete "
                                                                                           "Extensions...")
        self.__af.complete_extension()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Finding Grounded "
                                                                                           "Extension...")
        self.__result = self.__af.grounded_extension()
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Grounded Extension is: " +
              str(self.__result))

    def save_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Saving...")
        self.__af.save()

    def draw_argument_framework(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Drawing...")
        self.__af.draw()

    def result_tweets(self):
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Printing Grounded "
                                                                                           "Extension Tweets:")
        with open(self.__twitter.tweets_csv_path, mode="r", encoding="UTF-8") as file:
            for line in file:
                data = line.split("|")
                if int(data[0]) in self.__result:
                    print(data[0] + " | " + data[1] + " | " + data[5], end='\n')


def main(query, model, is_training):
    application = Main()
    application.setup_twitter_data(tweet_count=20, q=query)
    if is_training > 0:
        application.train_lstm_model(model_name=model)
    else:
        application.load_lstm_model(model_name=model)  # "RNN_vs200_b350_hs800_ml30"
    application.load_twitter_data()
    application.run_model_on_twitter()
    application.print_subsets()
    application.save_argument_framework()
    application.draw_argument_framework()
    application.result_tweets()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", nargs=1, type=int, default=0, help="0 - Load existing model. 1 - Perform "
                                                                         "training of model.", required=True)
    parser.add_argument("--modelname", nargs=1, type=str, default="RNN_vs200_b350_hs800_ml30", help="If training, "
                        "model name to save as, otherwise model name to load.", required=True)
    parser.add_argument("--query", nargs="*", type=str, help="Twitter search term.", required=True)
    args = parser.parse_args()
    main(query=args.query, model=args.modelname, is_training=args.training)
