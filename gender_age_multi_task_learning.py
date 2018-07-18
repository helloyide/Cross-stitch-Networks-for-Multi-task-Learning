import pickle
from datetime import datetime

import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
from keras.utils import to_categorical


def load_data():
    # train_X: (60000, 28, 28)
    # train_y: (60000,)
    # test_X: (10000, 28, 28)
    # test_y: (10000,)
    with open("saved_data", "rb") as file:
        # data is a list with length 2000
        # elements are {
        #   'image_path': str
        #   'gender': 'f'/'m'
        #   'age_young': bool
        #   'embedding': ndarray with shape (128,) dtype float64
        # }
        data = pickle.load(file)
    np.random.seed(1)
    np.random.shuffle(data)
    test = data[:200]
    train = data[200:]
    train_X = np.array([t["embedding"] for t in train])
    test_X = np.array([t["embedding"] for t in test])
    n_class_1 = 2
    train_y_1 = [0 if t["gender"] == 'f' else 1 for t in train]
    test_y_1 = [0 if t["gender"] == 'f' else 1 for t in test]
    n_class_2 = 2
    train_y_2 = [1 if t["age_young"] else 0 for t in train]
    test_y_2 = [1 if t["age_young"] else 0 for t in test]
    # train_X: (1800, 128)
    # test_X: (200, 128)
    # train_y: (1800, n_class)
    # test_y: (200, n_class)
    train_y_1 = to_categorical(train_y_1, n_class_1)
    test_y_1 = to_categorical(test_y_1, n_class_1)
    train_y_2 = to_categorical(train_y_2, n_class_2)
    test_y_2 = to_categorical(test_y_2, n_class_2)
    return train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2


def apply_cross_stitch(input1, input2):
    input1_reshaped = contrib.layers.flatten(input1)
    input2_reshaped = contrib.layers.flatten(input2)
    input = tf.concat((input1_reshaped, input2_reshaped), axis=1)

    # initialize with identity matrix
    cross_stitch = tf.get_variable("cross_stitch", shape=(input.shape[1], input.shape[1]), dtype=tf.float32,
                                   collections=['cross_stitches', tf.GraphKeys.GLOBAL_VARIABLES],
                                   initializer=tf.initializers.identity())
    output = tf.matmul(input, cross_stitch)

    # need to call .value to convert Dimension objects to normal value
    input1_shape = list(-1 if s.value is None else s.value for s in input1.shape)
    input2_shape = list(-1 if s.value is None else s.value for s in input2.shape)
    output1 = tf.reshape(output[:, :input1_reshaped.shape[1]], shape=input1_shape)
    output2 = tf.reshape(output[:, input1_reshaped.shape[1]:], shape=input2_shape)
    return output1, output2


def main(args):
    train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2 = load_data()

    m = train_X.shape[0]
    n_output_1 = test_y_1.shape[1]
    n_output_2 = test_y_2.shape[1]

    lr = args.lr
    n_epoch = args.n_epoch
    n_batch_size = args.n_batch_size
    reg_lambda = args.reg_lambda
    keep_prob = args.keep_prob
    cross_stitch_enabled = args.cross_stitch_enabled

    with tf.variable_scope("placeholder"):
        X = tf.placeholder(tf.float32, (None, 128), "X")
        y_1 = tf.placeholder(tf.float32, (None, n_output_1), "y_1")
        y_2 = tf.placeholder(tf.float32, (None, n_output_2), "y_2")
        is_training = tf.placeholder(tf.bool, (), "is_training")

    with tf.variable_scope("network"):
        with contrib.framework.arg_scope(
                [contrib.layers.fully_connected],
                # he initialization
                weights_initializer=contrib.layers.variance_scaling_initializer(),
                # l2 regularization
                weights_regularizer=contrib.layers.l2_regularizer(reg_lambda),
                # BN
                normalizer_fn=contrib.layers.batch_norm,
                normalizer_params={
                    "is_training": is_training,
                    "scale": True,
                    "updates_collections": None
                }
        ):
            fc1_1 = contrib.layers.fully_connected(X, 32, scope="fc1_1")
            fc1_2 = contrib.layers.fully_connected(X, 32, scope="fc1_2")

            if cross_stitch_enabled:
                with tf.variable_scope("cross_stitch_1"):
                    stitch1_1, stitch1_2 = apply_cross_stitch(fc1_1, fc1_2)
            else:
                stitch1_1, stitch1_2 = fc1_1, fc1_2

            fc2_1 = contrib.layers.fully_connected(stitch1_1, 32, scope="fc2_1")
            fc2_2 = contrib.layers.fully_connected(stitch1_2, 32, scope="fc2_2")

            if cross_stitch_enabled:
                with tf.variable_scope("cross_stitch_2"):
                    stitch2_1, stitch2_2 = apply_cross_stitch(fc2_1, fc2_2)
            else:
                stitch2_1, stitch2_2 = fc2_1, fc2_2

            dropout2_1 = contrib.layers.dropout(stitch2_1, keep_prob=keep_prob, is_training=is_training,
                                                scope="dropout2_1")
            dropout2_2 = contrib.layers.dropout(stitch2_2, keep_prob=keep_prob, is_training=is_training,
                                                scope="dropout2_2")

            fc3_1 = contrib.layers.fully_connected(dropout2_1, 32, scope="fc3_1")
            fc3_2 = contrib.layers.fully_connected(dropout2_2, 32, scope="fc3_2")

            if cross_stitch_enabled:
                with tf.variable_scope("cross_stitch_3"):
                    stitch3_1, stitch3_2 = apply_cross_stitch(fc3_1, fc3_2)
            else:
                stitch3_1, stitch3_2 = fc3_1, fc3_2

            dropout3_1 = contrib.layers.dropout(stitch3_1, keep_prob=keep_prob, is_training=is_training,
                                                scope="dropout3_1")
            dropout3_2 = contrib.layers.dropout(stitch3_2, keep_prob=keep_prob, is_training=is_training,
                                                scope="dropout3_2")

            output_1 = contrib.layers.fully_connected(dropout3_1, n_output_1, activation_fn=None, scope="output_1")
            output_2 = contrib.layers.fully_connected(dropout3_2, n_output_2, activation_fn=None, scope="output_2")

    with tf.variable_scope("loss"):
        loss_base_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1, logits=output_1))
        loss_base_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2, logits=output_2))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = loss_base_1 + loss_base_2 + tf.reduce_sum(reg_losses)

    with tf.variable_scope("evaluation"):
        accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output_1, axis=-1),
            tf.argmax(y_1, axis=-1)), tf.float32), name="accuracy_1")
        accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output_2, axis=-1),
            tf.argmax(y_2, axis=-1)), tf.float32), name="accuracy_2")
        accuracy = tf.divide(accuracy_1 + accuracy_2, 2.0, name="accuracy")

    with tf.variable_scope("train"):
        global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32, trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total, global_step=global_step)

    with tf.variable_scope("summary"):
        summary_loss_total = tf.summary.scalar("loss_total", loss_total)
        summary_accuracy_test = tf.summary.scalar("accuracy_test", accuracy)
        summary_accuracy_train = tf.summary.scalar("accuracy_train", accuracy)

    # standardization
    train_X_reshaped = train_X.reshape([train_X.shape[0], -1])
    train_X_means = np.mean(train_X_reshaped, axis=0, keepdims=True)
    train_X_stds = np.std(train_X_reshaped, axis=0, keepdims=True)

    def standardization(x):
        x_reshaped = x.reshape([x.shape[0], -1])
        result = (x_reshaped - train_X_means) / (train_X_stds + 1e-9)
        return result.reshape(x.shape)

    normalized_test_X = standardization(test_X)

    with tf.Session() as sess, tf.summary.FileWriter(
            "./tf_logs/fashion_minst_multi_task_learning/" + str(datetime.now().timestamp()),
            graph=tf.get_default_graph()) as f:
        sess.run(tf.global_variables_initializer())

        # similar logic as mnist's next_batch()
        epoch = 0
        index_in_epoch = 0
        while epoch < n_epoch:
            for _ in range(m // n_batch_size + 1):
                start = index_in_epoch
                if start + n_batch_size > m:
                    epoch += 1
                    n_rest_data = m - start
                    train_X_batch_rest = train_X[start:m]
                    train_y_batch_rest_1 = train_y_1[start:m]
                    train_y_batch_rest_2 = train_y_2[start:m]
                    # Shuffle train data
                    perm = np.arange(m)
                    np.random.shuffle(perm)
                    train_X = train_X[perm]
                    train_y_1 = train_y_1[perm]
                    train_y_2 = train_y_2[perm]
                    # Start next epoch
                    start = 0
                    index_in_epoch = n_batch_size - n_rest_data
                    end = index_in_epoch
                    train_X_batch_new = train_X[start:end]
                    train_y_batch_new_1 = train_y_1[start:end]
                    train_y_batch_new_2 = train_y_2[start:end]
                    # concatenate
                    train_X_batch = np.concatenate((train_X_batch_rest, train_X_batch_new), axis=0)
                    train_y_batch_1 = np.concatenate((train_y_batch_rest_1, train_y_batch_new_1), axis=0)
                    train_y_batch_2 = np.concatenate((train_y_batch_rest_2, train_y_batch_new_2), axis=0)
                else:
                    index_in_epoch += n_batch_size
                    end = index_in_epoch
                    train_X_batch = train_X[start:end]
                    train_y_batch_1 = train_y_1[start:end]
                    train_y_batch_2 = train_y_2[start:end]

                _, global_step_value, loss_total_value, summary_loss_total_value = \
                    sess.run([train_op, global_step, loss_total, summary_loss_total],
                             feed_dict={X: standardization(train_X_batch),
                                        y_1: train_y_batch_1,
                                        y_2: train_y_batch_2,
                                        is_training: True})

                if global_step_value % 100 == 0:
                    accuracy_train_value, summary_accuracy_train_value = \
                        sess.run([accuracy, summary_accuracy_train],
                                 feed_dict={X: standardization(train_X),
                                            y_1: train_y_1,
                                            y_2: train_y_2,
                                            is_training: False})
                    accuracy_test_value, summary_accuracy_test_value = \
                        sess.run([accuracy, summary_accuracy_test],
                                 feed_dict={X: normalized_test_X,
                                            y_1: test_y_1,
                                            y_2: test_y_2,
                                            is_training: False})

                    print(global_step_value, epoch, loss_total_value, accuracy_train_value, accuracy_test_value)
                    # cross_stitches = tf.get_collection("cross_stitches")
                    # print(cross_stitches[0].eval(sess))

                    f.add_summary(summary_loss_total_value, global_step=global_step_value)
                    f.add_summary(summary_accuracy_train_value, global_step=global_step_value)
                    f.add_summary(summary_accuracy_test_value, global_step=global_step_value)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, help="learning rate", default=0.0003)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=900)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=128)
    parser.add_argument("--reg_lambda", type=float, help="L2 regularization lambda", default=1e-4)
    parser.add_argument("--keep_prob", type=float, help="Dropout keep probability", default=0.8)
    parser.add_argument("--cross_stitch_enabled", type=bool, help="Use Cross Stitch or not", default=True)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
