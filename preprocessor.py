import numpy as np
import csv
import time
import datetime


class Preprocessor:
    def __init__(self, evidence_length, hypothesis_length, vector_size):
        self.__evidence_length = evidence_length
        self.__hypothesis_length = hypothesis_length
        self.__vector_size = vector_size
        self.__glove_word_map = {}

    def get_evidence_length(self):
        return self.__evidence_length

    def get_hypothesis_length(self):
        return self.__hypothesis_length

    def get_vector_size(self):
        return self.__vector_size

    def setup_word_map(self, file):
        with open(file, "r", encoding="utf-8") as glove:
            for line in glove:
                name, vector = tuple(line.split(" ", 1))
                self.__glove_word_map[name] = np.fromstring(vector, sep=" ")

    @staticmethod
    def setup_score(row):
        map_relation_to_val = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }
        values = np.zeros((3,))
        for num in range(1, 6):
            idx = row["label" + str(num)]
            if idx in map_relation_to_val:
                values[map_relation_to_val[idx]] += 1
        return values / (1.0 * np.sum(values))

    @staticmethod
    def fit_to_size(matrix, shape):
        res = np.zeros(shape, dtype=np.float16)
        slices = [slice(0, min(dim, shape[e])) for e, dim in enumerate(matrix.shape)]
        res[tuple(slices)] = matrix[tuple(slices)]
        return res

    def sent_to_seq(self, sentence):
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
        with open(file, "r") as dset:
            file = csv.DictReader(dset, delimiter='\t')
            evidences = []
            hypotheses = []
            scores = []
            for line in file:
                hypotheses.append(np.vstack(self.sent_to_seq(line["sentence1"].lower())[0]))
                evidences.append(np.vstack(self.sent_to_seq(line["sentence2"].lower())[0]))
                scores.append(self.setup_score(line))
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Stacking hypotheses...")
        hypotheses = np.stack([self.fit_to_size(x, (self.__hypothesis_length, self.__vector_size))
                               for x in hypotheses])

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": Stacking evidences...")
        evidences = np.stack([self.fit_to_size(x, (self.__evidence_length, self.__vector_size))
                              for x in evidences])

        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + ": update_data_scores: Done.")
        return (hypotheses, evidences), np.array(scores)
