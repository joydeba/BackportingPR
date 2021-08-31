'''
This model.py python file is part of ReBack, licensed under the CC0 1.0 Universal.
Details of the license can be found in the LICENSE file.
The current version of the ReBack can be always found at https://github.com/joydeba/BackportingPR
'''

import argparse
from Utils import extract_commit, reformat_commit_code, reformat_meta, reformat_discussion
from train import train_model
from predict import predict_model


def read_args():
    parser = argparse.ArgumentParser()
    # Arguments for training the model
    parser.add_argument('--train', action='store_true', help='training ReBack model')
    parser.add_argument('--data', type=str, default='./data/train_small.text',
                        help='the directory of our training data')

    # Arguments for testing the model
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Parameters for reformatting discussion and code
    parser.add_argument('--msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('--meta_length', type=int, default=512, help='the length of the meta')
    parser.add_argument('--code_hunk', type=int, default=8, help='the number of hunks in commit code')
    parser.add_argument('--code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Parameters for ReBack model
    parser.add_argument('--embedding_dim', type=int, default=32, help='the dimension of embedding vector')
    parser.add_argument('--filter_sizes', type=str, default='1, 2', help='the filter size of convolutional layers')
    parser.add_argument('--num_filters', type=int, default=32, help='the number of filters')
    parser.add_argument('--hidden_units', type=int, default=128, help='the number of nodes in hidden layers')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='dropout for training ReBack')
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='the number of epochs')
    parser.add_argument('--evaluate_every', type=int, default=500, help='evaluate model after this many steps')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='save model after this many steps')
    parser.add_argument('--num_checkpoints', type=int, default=100, help='the number of checkpoints to store')

    # Configaration arguments for tensorflow
    parser.add_argument('--allow_soft_placement', type=bool, default=True, help='allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')

    # Argumets for chosing the model itself
    parser.add_argument('--data_type', type=str, default='all', help='type of model for learning')
    parser.add_argument('--model', type=str, default='model', help='names of our model')
    return parser


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    commits = extract_commit(path_file=input_option.data)
    commits = reformat_discussion(commits)
    commits = reformat_meta(commits)
    commits = reformat_commit_code(commits=commits, num_file=5, num_hunk=input_option.code_hunk,
                                   num_loc=input_option.code_line, num_leng=input_option.code_length)

    if input_option.train is True:
        train_model(commits=commits, params=input_option)
        print '--------------------------------------------------------------------------------'
        print '--------------------------Model training is completed---------------------------'
        print '--------------------------------------------------------------------------------'
        exit()
    elif input_option.predict is True:
        predict_model(commits=commits, params=input_option)
        print '--------------------------------------------------------------------------------'
        print '--------------------------Model testing is completed----------------------------'
        print '--------------------------------------------------------------------------------'
        exit()
    else:
        print '--------------------------------------------------------------------------------'
        print 'Wrongs arguments are passed, please write -h to see the argument usage'
        print '--------------------------------------------------------------------------------'
        exit()
