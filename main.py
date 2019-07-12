from gather_tweets import TwitterMining
from run_RTE_training import TextualEntailment
from arg_framework import ArgFramework

import datetime
import time
import csv
import os


def main():
    af = ArgFramework()

    twitter = TwitterMining()
    twitter.mine_data(count=10000)

    train = False
    rte = TextualEntailment(is_training=train)
    if train:
        rte.run_training(save_as="rnn-128d-2048h-lr-0001-512b-ml-30")
        rte.run_validation()
        rte.run_test()

    # rnn-128d-1024h-out-10-lr-001-final
    model_path = os.getcwd() + '\\models\\rnn-840B-128d-1024h-final'
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Restoring model...")
    rte.load_model(model_path)

    sentences = []
    with open(twitter.tweets_csv_path, mode='r') as file:
        for line in file:
            split = line.split("|")
            sentences.append(split[-1])

    with open("lstm_results.csv", mode="w") as file:
        file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                res = rte.run_predict(sentences[j], sentences[i])
                if res in ["E", "C"]:
                    af.add_argument(sentences[j])
                    af.add_argument(sentences[i])
                    af.add_relation(sentences[j], sentences[i], 'r' if res in ["C"] else 'g')
                    file.writerow([j, sentences[j], res, i, sentences[i]])

    af.save()
    af.draw()

    # do analysis

    # draw it again


if __name__ == "__main__":
    main()
