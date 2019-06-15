import numpy as np
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker
import csv
import time
import datetime


class Params:
    def __init__(self):
        self.max_hypothesis_length = 30
        self.max_evidence_length = 30
        self.batch_size = 512
        self.hidden_size = 1024
        self.vector_size = 128
        self.n_classes = 3
        self.weight_decay = 0.95
        self.learning_rate = 0.001
        self.iterations = 5000000
        self.display_step = 100
        self.__glove_word_map = {}

    def setup_word_map(self, file):
        with open(file, "r", encoding="utf-8") as glove:
            for line in glove:
                name, vector = tuple(line.split(" ", 1))
                self.__glove_word_map[name] = np.fromstring(vector, sep=" ")

    def visualize(self, sentence):
        rows, words = self.sentence2sequence(sentence)
        mat = np.vstack(rows)

        fig = plot.figure()
        ax = fig.add_subplot(111)
        shown = ax.matshow(mat, aspect="auto")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.colorbar(shown)

        ax.set_yticklabels([""] + words)
        plot.show()

    @staticmethod
    def score_setup(row):
        convert_dict = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }
        score = np.zeros((3,))
        for x in range(1, 6):
            tag = row["label" + str(x)]
            if tag in convert_dict:
                score[convert_dict[tag]] += 1
        return score / (1.0 * np.sum(score))

    @staticmethod
    def fit_to_size(matrix, shape):
        res = np.zeros(shape, dtype=np.float16)
        slices = [slice(0, min(dim, shape[e])) for e, dim in enumerate(matrix.shape)]
        res[tuple(slices)] = matrix[tuple(slices)]
        return res

    def sentence2sequence(self, sentence):
        tokens = sentence.lower().split(" ")
        rows = []
        words = []
        for token in tokens:
            i = len(token)
            while len(token) > 0 and i > 0:
                word = token[:i]
                if word in self.__glove_word_map:
                    rows.append(self.__glove_word_map[word])
                    words.append(word)
                    token = token[i:]
                    i = len(token)
                else:
                    i = i - 1
        return rows, words

    def update_data_scores(self, file):
        with open(file, "r") as data:
            train = csv.DictReader(data, delimiter='\t')
            evi_sentences = []
            hyp_sentences = []
            scores = []
            for row in train:
                hyp_sentences.append(np.vstack(self.sentence2sequence(row["sentence1"].lower())[0]))
                evi_sentences.append(np.vstack(self.sentence2sequence(row["sentence2"].lower())[0]))
                scores.append(self.score_setup(row))
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Stacking hyp_sentences...")
        hyp_sentences = np.stack([self.fit_to_size(x, (self.max_hypothesis_length, self.vector_size))
                                  for x in hyp_sentences])

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Stacking evi_sentences...")
        evi_sentences = np.stack([self.fit_to_size(x, (self.max_evidence_length, self.vector_size))
                                  for x in evi_sentences])

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " update_data_scores: Done.")
        return (hyp_sentences, evi_sentences), np.array(scores)
