'''
This model.py python file is part of ReBack, licensed under the CC0 1.0 Universal.
Details of the license can be found in the LICENSE file.
The current version of the ReBack can be always found at https://github.com/joydeba/BackportingPR
'''

from padding import padding_commit
import tensorflow as tf
from ModelNet import ReBack 
import os
import datetime
from Utils import random_mini_batch, write_dict_file


def train_model(commits, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = \
        padding_commit(commits=commits, params=params)
    print 'Commit message dictionary has size: %i' % (len(dict_msg))
    print 'Commit code dictionary has size: %i' % (len(dict_code))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=params.allow_soft_placement,
                                      log_device_placement=params.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = ReBack(
                max_msg_length=params.msg_length,
                max_code_length=params.code_length,
                max_code_line=params.code_line,
                max_code_hunk=params.code_hunk,
                vocab_size_text=len(dict_msg),
                vocab_size_code=len(dict_code),
                embedding_size_text=params.embedding_dim,
                filter_sizes=list(map(int, params.filter_sizes.split(","))),
                num_filters=params.num_filters,
                l2_reg_lambda=params.l2_reg_lambda,
                num_classes=labels.shape[1],
                hidden_units=params.hidden_units)
            model.build_graph(model=params.data_type)

            # Defining training
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Defining output
            out_dir = os.path.abspath(os.path.join(os.path.curdir, params.model))
            print("Writing to {}\n".format(out_dir))
            write_dict_file(path_file=out_dir + '/dict_msg.txt', dictionary=dict_msg)
            write_dict_file(path_file=out_dir + '/dict_code.txt', dictionary=dict_code)

            # Loss and accuracy sumarization
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train sumarization
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


            checkpoint_dir = os.path.abspath(os.path.join(out_dir))
            checkpoint_prefix = os.path.join(checkpoint_dir, params.model)
            print "Checkpoints directory of our model %s" % (checkpoint_prefix)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=params.num_checkpoints)

            # Initializing variables
            sess.run(tf.global_variables_initializer())

            def train_step(input_msg, input_added_code, input_removed_code, input_labels):
                """
                Training step
                """
                feed_dict = {
                    model.input_msg: input_msg,
                    model.input_addedcode: input_added_code,
                    model.input_removedcode: input_removed_code,
                    model.input_y: input_labels,
                    model.dropout_keep_prob: params.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

        for i in xrange(0, params.num_epochs):
            # Batche creation
            mini_batches = random_mini_batch(X_msg=pad_msg, X_added_code=pad_added_code,
                                             X_removed_code=pad_removed_code, Y=labels,
                                             mini_batch_size=params.batch_size)
            for j in xrange(len(mini_batches)):
                batch = mini_batches[j]
                input_msg, input_added_code, input_removed_code, input_labels = batch
                train_step(input_msg, input_added_code, input_removed_code, input_labels)
                current_step = tf.train.global_step(sess, global_step)

            path_curr = saver.save(sess, checkpoint_prefix, global_step=current_step)
            path = saver.save(sess, checkpoint_prefix)
            print "Saving checkpoint at epoch %i to {}\n".format(path_curr) % i
