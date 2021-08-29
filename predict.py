'''
This model.py python file is part of ReBack, licensed under the CC0 1.0 Universal.
Details of the license can be found in the LICENSE file.
The current version of the ReBack can be always found at https://github.com/joydeba/BackportingPR
'''

from Utils import load_dict_file, mini_batches, write_file
from padding import padding_pred_commit
import os
import tensorflow as tf
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x_sum = np.sum(np.exp(x), axis=1)
    return np.exp(x) / e_x_sum[:, None]


def predict_model(commits, params):
    path_dict = os.path.abspath(os.path.join(os.path.curdir, params.model))
    dict_msg = load_dict_file(path_file=path_dict + '/dict_msg.txt')
    dict_meta = load_dict_file(path_file=path_dict + '/dict_meta.txt')
    dict_code = load_dict_file(path_file=path_dict + '/dict_code.txt')

    pad_msg, pad_meta, pad_added_code, pad_removed_code, labels = padding_pred_commit(commits=commits,
                                                                            params=params, dict_msg=dict_msg, dict_meta = dict_meta,
                                                                            dict_code=dict_code)
    checkpoint_dir = path_dict
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()     
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=params.allow_soft_placement,
            log_device_placement=params.log_device_placement)
        
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Loading saved meta graph and restoring variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            
            m_recall = tf.metrics.recall(tf.argmax(input_y, 1), predictions, name ="recall")
            m_f1_score = tf.contrib.metrics.f1_score(tf.argmax(input_y, 1), predictions, name ="f1_score")
            m_auc = tf.metrics.auc(tf.argmax(input_y, 1), predictions, name ="auc")
            m_precision = tf.metrics.precision(tf.argmax(input_y, 1), predictions, name ="precision")
                    

            # Geting placeholders from graph by name
            input_msg = graph.get_operation_by_name("input_msg").outputs[0]
            input_meta = graph.get_operation_by_name("input_meta").outputs[0]
            input_addedcode = graph.get_operation_by_name("input_addedcode").outputs[0]
            input_removedcode = graph.get_operation_by_name("input_removedcode").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Evaluating temsor
            scores = graph.get_operation_by_name("output/scores").outputs[0]
       

            # Batches for one epoch
            batches = mini_batches(X_msg=pad_msg, X_meta =pad_meta, X_added_code=pad_added_code,
                                   X_removed_code=pad_removed_code,
                                   Y=labels, mini_batch_size=params.batch_size)
                                                                                  
            commits_scores = list()
            accuracy_list = list()
            precision_list = list()
            recall_list = list()
            f1_score_list = list()
            auc_list = list()

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)    
            for batch in batches:
                
                batch_input_msg, batch_input_meta, batch_input_added_code, batch_input_removed_code, batch_input_labels = batch
                correct_predictions = tf.equal(predictions, tf.argmax(batch_input_labels, 1))
                m_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

                batch_scores, accuracy, precision, recall, f1_score, auc = sess.run([scores, m_accuracy, m_precision, m_recall, m_f1_score, m_auc],
                                        {input_msg: batch_input_msg, input_meta: batch_input_meta, input_addedcode: batch_input_added_code,
                                         input_removedcode: batch_input_removed_code, input_y:batch_input_labels, dropout_keep_prob: 1.0})
              

                batch_scores = np.ravel(softmax(batch_scores)[:, [1]])
                commits_scores = np.concatenate([commits_scores, batch_scores])


                accuracy_list = np.concatenate([accuracy_list, [accuracy]])
                precision_list = np.concatenate([precision_list, [precision[0]]])
                recall_list = np.concatenate([recall_list, [recall[0]]])
                f1_score_list = np.concatenate([f1_score_list, [f1_score[0]]])
                auc_list = np.concatenate([auc_list, [auc[0]]])

            print("acc {:g}, preci {}, reca {}, f1 {}, auc {}".format(np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list), np.mean(auc_list)))                     
            write_file(path_file=os.path.abspath(os.path.join(os.path.curdir)) + '/prediction.txt',
                       data=commits_scores)

