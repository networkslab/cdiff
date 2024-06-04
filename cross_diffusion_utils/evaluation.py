import torch


def add_eval_args(parser):

    # Train params
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--no-filter', dest='feature', action='store_false')
    parser.add_argument('--num_seqs_analysis', type=int, default=40)
