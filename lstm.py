from tqdm import tqdm

import tensorflow as tf
import numpy as np
import csv
import os


# noinspection PyUnusedLocal
class LSTM:
    def __init__(self, e_length, h_length, v_size):
        self.__max_evidence_length = e_length
        self.__max_hypothesis_length = h_length
        self.__batch_size = 256
        self.__hidden_size = 768
        self.__vector_size = v_size
        self.__n_classes = 3
        self.__weight_decay = 0.95
        self.__learning_rate = 0.001
        self.__iterations = 5000000
        self.__valid_iters = 147630
        self.__test_iters = 147360
        self.__display_step = 256
        self.__accuracy = None
        self.__loss = None
        self.__total_loss = None
        self.__training_accuracy = 0.0
        self.__training_loss = 0.0
        self.__validation_accuracy = 0.0
        self.__validation_loss = 0.0
        self.__testing_accuracy = 0.0
        self.__testing_loss = 0.0
        self.__log_file = os.getcwd() + "\\logs\\MODEL_TRAINING_RESULTS.csv"

        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.__hyp = tf.placeholder(tf.float32, [self.__batch_size, self.__max_hypothesis_length, self.__vector_size],
                                    'hypothesis')
        self.__evi = tf.placeholder(tf.float32, [self.__batch_size, self.__max_evidence_length, self.__vector_size],
                                    'evidence')
        self.__labels = tf.placeholder(tf.float32, [self.__batch_size, self.__n_classes], 'label')

        self.__input_keep = tf.placeholder_with_default(1.0, shape=())
        self.__output_keep = tf.placeholder_with_default(1.0, shape=())
        self.__state_keep = tf.placeholder_with_default(1.0, shape=())

        self.__lstm_back = tf.keras.layers.LSTMCell(self.__hidden_size)
        self.__lstm_drop_back = tf.contrib.rnn.DropoutWrapper(self.__lstm_back,
                                                              input_keep_prob=self.__input_keep,
                                                              output_keep_prob=self.__output_keep,
                                                              state_keep_prob=self.__state_keep)
        self.__lstm = tf.keras.layers.LSTMCell(self.__hidden_size)
        self._lstm_drop = tf.contrib.rnn.DropoutWrapper(self.__lstm,
                                                        input_keep_prob=self.__input_keep,
                                                        output_keep_prob=self.__output_keep,
                                                        state_keep_prob=self.__state_keep)

        self.__fc_initializer = tf.random_normal_initializer(stddev=0.1)
        self.__fc_weight = tf.get_variable('fc_weight', [2 * self.__hidden_size, self.__n_classes],
                                           initializer=self.__fc_initializer)
        self.__fc_bias = tf.get_variable('bias', [self.__n_classes])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.__fc_weight))

        self.__input = tf.concat([self.__hyp, self.__evi], 1)
        self.__input = tf.transpose(self.__input, [1, 0, 2])
        self.__input = tf.reshape(self.__input, [-1, self.__vector_size])
        self.__input = tf.split(self.__input, self.__max_hypothesis_length + self.__max_evidence_length, )

        self.__rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self.__lstm, self.__lstm_back,
                                                                           self.__input, dtype=tf.float32)
        self.__classification_scores = tf.matmul(self.__rnn_outputs[-1], self.__fc_weight) + self.__fc_bias

        with tf.variable_scope('Accuracy'):
            predicts = tf.cast(tf.argmax(self.__classification_scores, 1), 'int32')
            y_label = tf.cast(tf.argmax(self.__labels, 1), 'int32')
            corrects = tf.equal(predicts, y_label)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            self.__accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        with tf.variable_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__classification_scores,
                                                                       labels=self.__labels)
            self.__loss = tf.reduce_mean(cross_entropy)
            self.__total_loss = self.__loss + self.__weight_decay * tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))

        self.__optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(self.__total_loss)
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.__sess.close()

    def load_model(self, path):
        tf.train.Saver().restore(self.__sess, path)

    def get_evi_length(self):
        return self.__max_evidence_length

    def get_hyp_length(self):
        return self.__max_hypothesis_length

    def get_vector_size(self):
        return self.__vector_size

    def train(self, df_list, c_scores, save=None):
        training_iterations = range(0, self.__iterations, self.__batch_size)
        training_iterations = tqdm(training_iterations)

        avg_acc = 0.0
        avg_loss = 0.0

        with tf.device("/device:GPU:0"):
            for i in training_iterations:
                batch = np.random.randint(df_list[0].shape[0], size=self.__batch_size)
                hyps = df_list[0][batch, :]
                evis = df_list[1][batch, :]
                labels = c_scores[batch]

                self.__sess.run([self.__optimizer], feed_dict={self.__hyp: hyps,
                                                               self.__evi: evis,
                                                               self.__labels: labels,
                                                               self.__input_keep: 0.1})
                if (i / self.__batch_size) % self.__display_step == 0 and i != 0:
                    acc = self.__sess.run(self.__accuracy, feed_dict={self.__hyp: hyps,
                                                                      self.__evi: evis,
                                                                      self.__labels: labels,
                                                                      self.__input_keep: 0.1})
                    tmp_loss = self.__sess.run(self.__loss, feed_dict={self.__hyp: hyps,
                                                                       self.__evi: evis,
                                                                       self.__labels: labels,
                                                                       self.__input_keep: 0.1})
                    avg_acc = avg_acc + acc
                    avg_loss = avg_loss + tmp_loss
                    print("Iter " + str(i / self.__batch_size) + ", Minibatch Loss = " + "{:.5f}".format(tmp_loss) +
                          ", Training Accuracy = " + "{:.5f}".format(acc))

            self.__training_loss = avg_loss / len(training_iterations.iterable) * self.__display_step
            self.__training_accuracy = avg_acc / len(training_iterations.iterable) * self.__display_step
            print("------------------------------------------------------------------------------")
            print("Training Minibatch Loss = " + "{:.5f}".format(self.__training_loss) +
                  ", Training Accuracy = " + "{:.5f}".format(self.__training_accuracy))
            print("------------------------------------------------------------------------------")

        if save is not None:
            saver = tf.train.Saver()
            saver.save(self.__sess, os.getcwd() + '\\models\\' + save)

    def validate(self, df_list, c_scores):
        validation_iterations = range(0, self.__valid_iters, self.__batch_size)
        validation_iterations = tqdm(validation_iterations)

        acc = 0.0
        tmp_loss = 0.0

        with tf.device("/device:GPU:0"):
            for _ in validation_iterations:
                batch = np.random.randint(df_list[0].shape[0], size=self.__batch_size)
                hyps = df_list[0][batch, :]
                evis = df_list[1][batch, :]
                labels = c_scores[batch]

                self.__sess.run([self.__classification_scores], feed_dict={self.__hyp: hyps,
                                                                           self.__evi: evis,
                                                                           self.__labels: labels})
                acc = acc + self.__sess.run(self.__accuracy, feed_dict={self.__hyp: hyps,
                                                                        self.__evi: evis,
                                                                        self.__labels: labels})
                tmp_loss = tmp_loss + self.__sess.run(self.__total_loss, feed_dict={self.__hyp: hyps,
                                                                                    self.__evi: evis,
                                                                                    self.__labels: labels})

            self.__validation_loss = tmp_loss / len(validation_iterations.iterable)
            self.__validation_accuracy = acc / len(validation_iterations.iterable)
            print("Validation Minibatch Loss = " + "{:.5f}".format(self.__validation_loss) +
                  ", Validation Accuracy = " + "{:.5f}".format(self.__validation_accuracy))
            print("------------------------------------------------------------------------------")

    def test(self, df_list, c_scores):
        testing_iterations = range(0, self.__test_iters, self.__batch_size)
        testing_iterations = tqdm(testing_iterations)

        acc = 0.0
        tmp_loss = 0.0

        with tf.device("/device:GPU:0"):
            for _ in testing_iterations:
                batch = np.random.randint(df_list[0].shape[0], size=self.__batch_size)
                hyps = df_list[0][batch, :]
                evis = df_list[1][batch, :]
                labels = c_scores[batch]

                self.__sess.run([self.__classification_scores], feed_dict={self.__hyp: hyps,
                                                                           self.__evi: evis,
                                                                           self.__labels: labels})
                acc = acc + self.__sess.run(self.__accuracy, feed_dict={self.__hyp: hyps,
                                                                        self.__evi: evis,
                                                                        self.__labels: labels})
                tmp_loss = tmp_loss + self.__sess.run(self.__loss, feed_dict={self.__hyp: hyps,
                                                                              self.__evi: evis,
                                                                              self.__labels: labels})

            self.__testing_loss = tmp_loss / len(testing_iterations.iterable)
            self.__testing_accuracy = acc / len(testing_iterations.iterable)
            print("Testing Minibatch Loss = " + "{:.5f}".format(self.__testing_loss) +
                  ", Testing Accuracy = " + "{:.5f}".format(self.__testing_accuracy))
            print("------------------------------------------------------------------------------")

    def log_results(self):
        if not os.path.isfile(self.__log_file):
            with open(self.__log_file, mode='w', encoding='utf-8', newline='') as file:
                file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                file.writerow(["MAX_E_LENGTH", "MAX_H_LENGTH", "BATCH_SIZE", "HIDDEN_SIZE", "VECTOR_SIZE", "N_CLASSES",
                               "WEIGHT_DECAY", "LEARNING_RATE", "TRAIN_ITERS", "VALID_ITERS", "TEST_ITERS",
                               "DISPLAY_STEP", "TRAIN_ACC", "TRAIN_LOSS", "VALID_ACC", "VALID_LOSS", "TEST_ACC",
                               "TEST_LOSS"])
        with open(self.__log_file, mode='a', encoding='utf-8', newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file.writerow([self.__max_evidence_length, self.__max_hypothesis_length, self.__batch_size,
                           self.__hidden_size, self.__vector_size, self.__n_classes, self.__weight_decay,
                           self.__learning_rate, self.__iterations, self.__valid_iters, self.__test_iters,
                           self.__display_step, self.__training_accuracy, self.__training_loss,
                           self.__validation_accuracy, self.__validation_loss, self.__testing_accuracy,
                           self.__testing_loss])

    def run_prediction(self, evi_sentence, hyp_sentence):
        with tf.device("/device:GPU:0"):
            prediction = self.__sess.run(self.__classification_scores,
                                         feed_dict={self.__hyp: (evi_sentence * self.__batch_size),
                                                    self.__evi: (hyp_sentence * self.__batch_size),
                                                    self.__labels: [[0, 0, 0]] * self.__batch_size})

        return ["E", "N", "C"][np.argmax(prediction[0])]
