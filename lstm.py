from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os


class LSTM:
    def __init__(self, model):
        self.__model = model
        self.__hyp = tf.placeholder(tf.float32,
                                    [model.batch_size, model.max_hypothesis_length, model.vector_size],
                                    'hypothesis')
        self.__evi = tf.placeholder(tf.float32,
                                    [model.batch_size, model.max_evidence_length, model.vector_size],
                                    'evidence')
        self.__y = tf.placeholder(tf.float32,
                                  [model.batch_size, model.n_classes],
                                  'label')

        self.__lstm_back = tf.keras.layers.LSTMCell(model.hidden_size)
        self.__lstm_drop_back = tf.contrib.rnn.DropoutWrapper(self.__lstm_back,
                                                              model.input_keep,
                                                              model.output_keep)
        self.__lstm = tf.keras.layers.LSTMCell(model.hidden_size)
        self._lstm_drop = tf.contrib.rnn.DropoutWrapper(self.__lstm,
                                                        model.input_keep,
                                                        model.output_keep)

        self.__fc_initializer = tf.random_normal_initializer(stddev=0.1)
        self.__fc_weight = tf.get_variable('fc_weight',
                                           [2 * model.hidden_size, model.n_classes],
                                           initializer=self.__fc_initializer)
        self.__fc_bias = tf.get_variable('bias',
                                         [model.n_classes])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             tf.nn.l2_loss(self.__fc_weight))

        self.__input = tf.concat([self.__hyp, self.__evi], 1)
        self.__input = tf.transpose(self.__input, [1, 0, 2])
        self.__input = tf.reshape(self.__input, [-1, model.vector_size])
        self.__input = tf.split(self.__input, model.max_hypothesis_length + model.max_evidence_length, )

        self.__rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self.__lstm,
                                                                           self.__lstm_back,
                                                                           self.__input,
                                                                           dtype=tf.float32)
        self.__classification_scores = tf.matmul(self.__rnn_outputs[-1], self.__fc_weight) + self.__fc_bias
        self.__accuracy = None
        self.__loss = None
        self.__total_loss = None

        self.valid_iters = 20000
        self.test_iters = 20000

    def setup_accuracy_scope(self):
        with tf.variable_scope('Accuracy'):
            predicts = tf.cast(tf.argmax(self.__classification_scores, 1), 'int32')
            y_label = tf.cast(tf.argmax(self.__y, 1), 'int32')
            corrects = tf.equal(predicts, y_label)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            self.__accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    def setup_loss_scope(self):
        with tf.variable_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__classification_scores,
                                                                       labels=self.__y)
            self.__loss = tf.reduce_mean(cross_entropy)
            self.__total_loss = self.__loss + self.__model.weight_decay * \
                                tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    def train(self, data_feature_list, correct_scores, save=None):
        optimizer = tf.train.AdamOptimizer(self.__model.learning_rate).minimize(self.__total_loss)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        training_iterations = range(0, self.__model.iterations, self.__model.batch_size)
        training_iterations = tqdm(training_iterations)

        avg_acc = 0.0
        avg_loss = 0.0

        with tf.device("/device:GPU:0"):
            for i in training_iterations:
                batch = np.random.randint(data_feature_list[0].shape[0],
                                          size=self.__model.batch_size)
                hyps = data_feature_list[0][batch, :]
                evis = data_feature_list[1][batch, :]
                ys = correct_scores[batch]

                sess.run([optimizer], feed_dict={self.__hyp: hyps,
                                                 self.__evi: evis,
                                                 self.__y: ys})
                if (i / self.__model.batch_size) % self.__model.display_step == 0:
                    acc = sess.run(self.__accuracy, feed_dict={self.__hyp: hyps,
                                                               self.__evi: evis,
                                                               self.__y: ys})
                    tmp_loss = sess.run(self.__loss, feed_dict={self.__hyp: hyps,
                                                                self.__evi: evis,
                                                                self.__y: ys})
                    avg_acc = avg_acc + acc
                    avg_loss = avg_loss + tmp_loss
                    print("Iter " + str(i / self.__model.batch_size) +
                          ", Minibatch Loss = " + "{:.6f}".format(tmp_loss) +
                          ", Training Accuracy = " + "{:.5f}".format(acc))

            print("------------------------------------------------------------------------------")
            print("Training Minibatch Loss = " + "{:.6f}".format(tmp_loss / len(training_iterations.iterable) *
                                                                 self.__model.display_step) +
                  ", Training Accuracy = " + "{:.5f}".format(avg_acc / len(training_iterations.iterable) *
                                                             self.__model.display_step))
            print("------------------------------------------------------------------------------")

        if save is not None:
            saver = tf.train.Saver()
            saver.save(sess, os.getcwd() + '\\models\\' + save)

        return sess

    def validate(self, data_feature_list, correct_scores, sess):
        validation_iterations = range(0, self.valid_iters, self.__model.batch_size)
        validation_iterations = tqdm(validation_iterations)
        acc = 0.0
        tmp_loss = 0.0

        with tf.device("/device:GPU:0"):
            for _ in validation_iterations:
                batch = np.random.randint(data_feature_list[0].shape[0],
                                          size=self.__model.batch_size)
                hyps = data_feature_list[0][batch, :]
                evis = data_feature_list[1][batch, :]
                ys = correct_scores[batch]

                sess.run([self.__classification_scores],
                         feed_dict={self.__hyp: hyps,
                                    self.__evi: evis,
                                    self.__y: ys})
                acc = acc + sess.run(self.__accuracy,
                                     feed_dict={self.__hyp: hyps,
                                                self.__evi: evis,
                                                self.__y: ys})
                tmp_loss = tmp_loss + sess.run(self.__total_loss,
                                               feed_dict={self.__hyp: hyps,
                                                          self.__evi: evis,
                                                          self.__y: ys})

            print("Validation Minibatch Loss = " + "{:.6f}".format(tmp_loss / len(validation_iterations.iterable)) +
                  ", Validation Accuracy = " + "{:.5f}".format(acc / len(validation_iterations.iterable)))
            print("------------------------------------------------------------------------------")

    def test(self, data_feature_list, correct_scores, sess):
        testing_iterations = range(0, self.test_iters, self.__model.batch_size)
        testing_iterations = tqdm(testing_iterations)
        acc = 0.0
        tmp_loss = 0.0

        with tf.device("/device:GPU:0"):
            for _ in testing_iterations:
                batch = np.random.randint(data_feature_list[0].shape[0],
                                          size=self.__model.batch_size)
                hyps = data_feature_list[0][batch, :]
                evis = data_feature_list[1][batch, :]
                ys = correct_scores[batch]

                sess.run([self.__classification_scores],
                         feed_dict={self.__hyp: hyps,
                                    self.__evi: evis,
                                    self.__y: ys})
                acc = acc + sess.run(self.__accuracy,
                                     feed_dict={self.__hyp: hyps,
                                                self.__evi: evis,
                                                self.__y: ys})
                tmp_loss = tmp_loss + sess.run(self.__loss,
                                               feed_dict={self.__hyp: hyps,
                                                          self.__evi: evis,
                                                          self.__y: ys})

            print("Testing Minibatch Loss = " + "{:.6f}".format(tmp_loss / len(testing_iterations.iterable)) +
                  ", Testing Accuracy = " + "{:.5f}".format(acc / len(testing_iterations.iterable)))
            print("------------------------------------------------------------------------------")