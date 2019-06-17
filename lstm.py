from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os


# noinspection PyUnusedLocal
class LSTM:
    def __init__(self, e_length, h_length, v_size):
        self.__max_hypothesis_length = h_length
        self.__max_evidence_length = e_length
        self.__batch_size = 512
        self.__hidden_size = 1024
        self.__vector_size = v_size
        self.__n_classes = 3
        self.__weight_decay = 0.95
        self.__learning_rate = 0.01
        self.__iterations = 5000000
        self.__display_step = 100
        self.__valid_iters = 100000
        self.__test_iters = 100000
        self.__accuracy = None
        self.__loss = None
        self.__total_loss = None

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
        print("TF Session Ended.")
        self.__sess.close()

    def load_session(self, path):
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

            print("------------------------------------------------------------------------------")
            print("Training Minibatch Loss = " + "{:.5f}".format(avg_loss / len(training_iterations.iterable) *
                                                                 self.__display_step) +
                  ", Training Accuracy = " + "{:.5f}".format(avg_acc / len(training_iterations.iterable) *
                                                             self.__display_step))
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

            print("Validation Minibatch Loss = " + "{:.6f}".format(tmp_loss / len(validation_iterations.iterable)) +
                  ", Validation Accuracy = " + "{:.5f}".format(acc / len(validation_iterations.iterable)))
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

            print("Testing Minibatch Loss = " + "{:.6f}".format(tmp_loss / len(testing_iterations.iterable)) +
                  ", Testing Accuracy = " + "{:.5f}".format(acc / len(testing_iterations.iterable)))
            print("------------------------------------------------------------------------------")

    def run_textual_entailment(self, evi_sentence, hyp_sentence):
        with tf.device("/device:GPU:0"):
            prediction = self.__sess.run(self.__classification_scores,
                                         feed_dict={self.__hyp: (evi_sentence * self.__batch_size),
                                                    self.__evi: (hyp_sentence * self.__batch_size),
                                                    self.__labels: [[0, 0, 0]] * self.__batch_size})

        return ["E", "N", "C"][np.argmax(prediction[0])]
