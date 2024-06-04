import torch
from torch.utils.data import DataLoader, ConcatDataset

# from tpp_diffusion.datasets.dataset import SeqDataset, collate
from tpp_utils_seq2seq.dataset_seq2seq.dataset_boxcox import SeqDatasetBoxCox\
    , collateboxcox, load_dataset_boxcox
from tpp_utils_seq2seq.dataset_seq2seq.dataset_ln import SeqDatasetLn\
    , collateln, load_dataset_ln
import os
# import parser
import argparse

dataset_choices = {'stackoverflow', 'taxi', 'taobao', 'syn_5_0_2', 'financial', 'retweet', 'amazon', 'mooc', 'lastfm'}



def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='stackoverflow', choices=dataset_choices)
    parser.add_argument('--dataset_dir', type=str, default='data/taxi/')
    parser.add_argument('--validation', type=eval, default=True)
    parser.add_argument('--tgt_len', type=int, default=20)
    parser.add_argument('--boxcox', action='store_true', help="if not boxcox then ln")
    parser.add_argument('--no-boxcox', dest='feature', action='store_false')

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=eval, default=False)


def get_data_id(args):
    return args.dataset


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.boxcox:
        print('loading {} datasets with boxcox preprocessing...'.format('train'))
        # train = SeqDatasetArBoxCox(dataset_dir=args.dataset_dir,
        #                    mode='train', target_length=args.tgt_len, device=args.device, data_name=args.dataset)
        train_loader, train = load_dataset_boxcox(dataset_dir=args.dataset_dir, mode='train',
                                                 device=args.device, data_name=args.dataset,
                                                  target_length=args.tgt_len)
        lmbda_boxcox = train.fitted_lambda
        scale = train.scale
        train_mean = train.mean_inter_time
        train_std = train.std_inter_time
        train_min = train.min_inter_time
        train_bc_mean = train.boxcox_mean
        train_bc_std = train.boxcox_std
        train_bc_min = train.boxcox_min_inter_time
        args.train_lambda_boxcox = lmbda_boxcox
        args.scale = scale
        args.train_mean = train_mean
        args.train_std = train_std
        args.train_min = train_min
        args.train_bc_mean = train_bc_mean
        args.train_bc_std = train_bc_std
        args.train_bc_min = train_bc_min
        args.min_inter_time = train_min

        print('loading {} datasets with boxcox preprocessing...'.format('val'))
        val_loader, valid = load_dataset_boxcox(dataset_dir=args.dataset_dir, mode='dev', lmbda_boxcox=lmbda_boxcox,
                                                   scale=scale, train_mean=train_mean, train_std=train_std,
                                                   train_min=train_min, train_bc_mean=train_bc_mean,
                                                   train_bc_std=train_bc_std,
                                                   train_bc_min=train_bc_min, target_length=args.tgt_len,
                                                   data_name=args.dataset)
        print('loading {} datasets with boxcox preprocessing...'.format('test'))
        test_loader, test = load_dataset_boxcox(dataset_dir=args.dataset_dir, mode='test', lmbda_boxcox=lmbda_boxcox,
                                                   scale=scale, train_mean=train_mean, train_std=train_std,
                                                   train_min=train_min, train_bc_mean=train_bc_mean,
                                                   train_bc_std=train_bc_std,
                                                   train_bc_min=train_bc_min, target_length=args.tgt_len,
                                                   data_name=args.dataset)

    else:
        print('loading {} datasets with log preprocessing...'.format('train'))

        train_loader, train = load_dataset_ln(dataset_dir=args.dataset_dir, mode='train',
                                                 device=args.device, data_name=args.dataset, target_length=args.tgt_len)

        train_mean = train.mean_inter_time
        train_std = train.std_inter_time
        train_min = train.min_inter_time
        train_ln_mean = train.ln_mean
        train_ln_std = train.ln_std
        train_ln_min = train.ln_min_inter_time

        args.train_mean = train_mean
        args.train_std = train_std
        args.train_min = train_min
        args.train_ln_mean = train_ln_mean
        args.train_ln_std = train_ln_std
        args.train_ln_min = train_ln_min
        args.min_inter_time = train_min

        print('loading {} datasets with log preprocessing...'.format('val'))
        val_loader, valid = load_dataset_ln(dataset_dir=args.dataset_dir, mode='dev',
                                               train_mean=train_mean, train_std=train_std,
                                               train_min=train_min, train_bc_mean=train_ln_mean,
                                               train_bc_std=train_ln_std,
                                               train_bc_min=train_ln_min, target_length=args.tgt_len)
        print('loading {} datasets with log preprocessing...'.format('test'))
        test_loader, test = load_dataset_ln(dataset_dir=args.dataset_dir, mode='test',
                                               train_mean=train_mean, train_std=train_std,
                                               train_min=train_min, train_bc_mean=train_ln_mean,
                                               train_bc_std=train_ln_std,
                                               train_bc_min=train_ln_min, target_length=args.tgt_len)
    data_shape = (args.tgt_len,)


    if args.dataset == 'amazon':
        args.num_classes = 16
        if args.tgt_len == 25:
            args.time_range = 13
        if args.tgt_len == 20:
            args.time_range = 13
        if args.tgt_len == 10:
            args.time_range = 7
        if args.tgt_len == 5:
            args.time_range = 4


    if args.dataset == 'stackoverflow':
        args.num_classes = 22
        if args.tgt_len == 25:
            args.time_range = 13
        if args.tgt_len == 20:
            args.time_range = 20
        if args.tgt_len == 10:
            args.time_range = 20/2
        if args.tgt_len == 5:
            args.time_range = 20/4
    # Dataset
    if args.dataset == 'taxi':
        args.num_classes = 10
        # args.time_range = 4.5
        if args.tgt_len == 25:
            args.time_range = 4.5
        if args.tgt_len == 20:
            args.time_range = 4.5
        if args.tgt_len == 10:
            args.time_range = 4.5/2
        if args.tgt_len == 5:
            args.time_range = 4.5/4
    if args.dataset == 'taobao':
        args.num_classes = 17
        if args.tgt_len == 25:
            args.time_range = 6.5/3
        # args.time_range = 1.5
        if args.tgt_len == 20:
            args.time_range = 6.5
        if args.tgt_len == 10:
            args.time_range = 6.5/2
        if args.tgt_len == 5:
            args.time_range = 6.5/4
    if args.dataset == 'syn_5_0_2':
        args.num_classes = 5
        # args.time_range = 2
        if args.tgt_len == 25:
            args.time_range = 2
        if args.tgt_len == 20:
            args.time_range = 2
        if args.tgt_len == 10:
            args.time_range = 2/2
        if args.tgt_len == 5:
            args.time_range = 2/4
    if args.dataset == 'retweet':
        args.num_classes = 3
        if args.tgt_len == 25:
            args.time_range = 500
        if args.tgt_len == 20:
            args.time_range = 500
        if args.tgt_len == 10:
            args.time_range = 250
        if args.tgt_len == 5:
            args.time_range = 150
    args.mean_inter_time = train.mean_inter_time
    args.std_inter_time = train.std_inter_time
    args.min_inter_time = train.min_inter_time
    # Data Loader
    if args.boxcox:
        if args.validation:
            train_loader = DataLoader(train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collateboxcox)
            # eval_loader = DataLoader(valid, batch_size=args.batch_size,
            #                          shuffle=False, collate_fn=collateboxcox)
            eval_loader = DataLoader(valid, batch_size=1024,
                                     shuffle=False, collate_fn=collateboxcox)
        else:
            dataset_train = ConcatDataset([train, valid])
            train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collateboxcox)
            eval_loader = DataLoader(test, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collateboxcox)
    else:
        if args.validation:
            train_loader = DataLoader(train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collateln)
            # eval_loader = DataLoader(valid, batch_size=args.batch_size,
            #                          shuffle=False, collate_fn=collateln)
            eval_loader = DataLoader(valid, batch_size=1024,
                                     shuffle=False, collate_fn=collateln)
        else:
            dataset_train = ConcatDataset([train, valid])
            train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collateln)
            eval_loader = DataLoader(test, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collateln)

    num_classes = args.num_classes
    return train_loader, eval_loader, data_shape, num_classes
