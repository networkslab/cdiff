# import sys
# sys.path.append('../')
import os
import torch
import argparse
import pickle
import time
import numpy as np
from prettytable import PrettyTable
import matplotlib.cm as cm
import cross_diffusion_utils
from cross_diffusion_utils.utils import add_parent_path, set_seeds

# Exp
from experiment import Experiment, add_exp_args
import warnings
warnings.filterwarnings("ignore")
# Data
add_parent_path(level=1)
from tpp_utils_seq2seq.dataset_seq2seq.data import get_data, get_data_id, add_data_args

# Model
# from tab_event_seq_model import get_model, get_model_id, add_model_args
from tab_event_seq_model import get_model, get_model_id, add_model_args

# Optim
from cross_diffusion_utils.expdecay import get_optim, get_optim_id, add_optim_args

# Eval
from cross_diffusion_utils.evaluation import add_eval_args

from tpp_utils_seq2seq.dataset_seq2seq.dataset_ln import SeqDatasetLn, load_dataset_ln, collateln
from tpp_utils_seq2seq.dataset_seq2seq.dataset_boxcox import SeqDatasetBoxCox, load_dataset_boxcox, collateboxcox

# Metric
from metrics import distance_between_event_seq, time_rmse_tensor, mape_tensor, sMape_tensor, filter_points
from metrics import get_distances_diffusion, type_rmse_diffusion, rmse_mae_num_events_diffusion

import tpp_utils_seq2seq.dataset_seq2seq.Constants as Constants
import scipy
import argparse
from scipy.special import boxcox, inv_boxcox


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def save_args(args):
    # Save args
    with open(os.path.join(args.log_path, 'args.pickle'), "wb") as f:
        pickle.dump(args, f)

    # Save args table
    args_table = get_args_table(vars(args))
    with open(os.path.join(args.log_path, 'args_table.txt'), "w") as f:
        f.write(str(args_table))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    add_exp_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_optim_args(parser)
    add_eval_args(parser)
    return parser.parse_args()


def run_train(args):
    set_seeds(args.seed)

    ##################
    ## Specify data ##
    ##################

    train_loader, eval_loader, data_shape, num_classes = get_data(args)
    data_id = get_data_id(args)

    ###################
    ## Specify model ##
    ###################

    model = get_model(args, num_classes=num_classes)
    model_id = get_model_id(args)

    #######################
    ## Specify optimizer ##
    #######################

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    optim_id = get_optim_id(args)

    ##############
    ## Training ##
    ##############

    args.validation = True
    exp = Experiment(args=args,
                     data_id=data_id,
                     model_id=model_id,
                     optim_id=optim_id,
                     train_loader=train_loader,
                     eval_loader=eval_loader,
                     model=model,
                     optimizer=optimizer,
                     scheduler_iter=scheduler_iter,
                     scheduler_epoch=scheduler_epoch)

    exp.run()

    return args



def run_eval(args):
    #############################################
    ################# Load args #################
    #############################################
    eval_seed = 0
    if args == None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_path', type=str, default='./')
        parser.add_argument('--eval_seed', type=int, default=0)
        # we pick 24, 8, 81, 21, 23, dont ask me why. Search them online if you dont know. idiot
        args = parser.parse_args()
        eval_seed = args.eval_seed

    path_args = '{}/args.pickle'.format(args.log_path)
    path_check = '{}/check/checkpoint.pt'.format(args.log_path)

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    assert args.tgt_len is not None, 'Currently, length has to be specified.'
    if eval_seed == 0:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(eval_seed)

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    args.num_timesteps = args.diffusion_steps
    num_samples = args.num_samples

    distance_del_cost = [0.05, 0.5, 1, 1.5, 2, 3, 4]
    trans_cost = 1.0
    args.distance_del_cost = distance_del_cost
    args.trans_cost = trans_cost

    ###################################################
    ################## Load dataset ###################
    ###################################################
    # Dataset
    if args.boxcox:
        train_loader, train = load_dataset_boxcox(dataset_dir=args.dataset_dir, mode='train',
                                                  device=args.device, data_name=args.dataset,
                                                  target_length=args.tgt_len)
    else:
        train_loader, train = load_dataset_ln(dataset_dir=args.dataset_dir, mode='train',
                                              device=args.device, data_name=args.dataset, target_length=args.tgt_len)

    std_inter_time = train.std_inter_time
    mean_inter_time = train.mean_inter_time
    min_inter_time = train.min_inter_time

    args.validation = False

    train_loader, test_loader, data_shape, num_classes = get_data(args)

    args.validation = True

    #########################################################
    ##################### Specify model #####################
    #########################################################

    model = get_model(args, num_classes=num_classes)
    checkpoint = torch.load(path_check)
    model.load_state_dict(checkpoint['model'])
    print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params: } training parameters.')

    ##########################################################################################
    ##################################### Specify Saving #####################################
    ##########################################################################################

    ############## Saving base ##############

    path_samples = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                )

    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    args.path_samples = path_samples

    ############## Result log ##############

    path_samples_result = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/result.txt'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                       )

    if not os.path.exists(os.path.dirname(path_samples_result)):
        os.mkdir(os.path.dirname(path_samples_result))

    args.path_samples_result = path_samples_result

    ############## dt Samples Saving Path ##############

    path_samples_dt = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/samples_dt.pt'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                   )

    args.path_samples_dt = path_samples_dt

    path_samples_chain_dt = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/samples_chain_dt.pt'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                   )
    args.path_samples_chain_dt = path_samples_chain_dt

    ############## type Samples Saving Path ##############

    path_samples_type = os.path.join(args.log_path,
                                     'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/samples_type.pt'.format(
                                         checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                     )

    args.path_samples_type = path_samples_type

    path_samples_chain_type = os.path.join(args.log_path,
                                     'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/samples_type.pt'.format(
                                         checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                     )

    args.path_samples_chain_type = path_samples_chain_type

    ############## dt ground truth Saving Path ##############

    path_gt_dt = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/gt_dt.pt'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                              )

    args.path_gt_dt = path_gt_dt

    ############## type ground truth Saving Path ##############

    path_gt_type = os.path.join(args.log_path, 'samples/sample_ep{}_s{}_num_s_{}_num_steps_{}/gt_type.pt'.format(
        checkpoint['current_epoch'], args.seed, args.num_samples, args.num_timesteps)
                                )

    args.path_gt_type = path_gt_type

    ####################################################################################
    ##################################### Sampling #####################################
    ####################################################################################

    device = args.device
    model = model.to(device)
    model = model.eval()
    # if args.double: model = model.double()

    pred_e_total = torch.empty(0, args.tgt_len, num_samples).to('cpu')
    pred_x_total = torch.empty(0, args.tgt_len, num_samples).to('cpu')
    gt_e_total = torch.empty(0, args.tgt_len).to('cpu')
    gt_x_total = torch.empty(0, args.tgt_len).to('cpu')

    with torch.no_grad():
        since = time.time()
        for iteration, batch in enumerate(test_loader):

            history_times = batch.history_times
            hist_e = batch.history_types.long()
            hist_x = batch.history_dt
            target_times = batch.target_times
            tgt_e = batch.target_types.long()
            tgt_x = batch.target_dt
            target_onehots = batch.target_onehots
            unnormed_history_dt = batch.unnormed_history_dt
            unnormed_target_dt = batch.unnormed_target_dt

            num_elem = tgt_e.flatten().size(0)

            pred_e = torch.empty(tgt_e.size(0), tgt_e.size(1), 0).to(device)
            pred_x = torch.empty(tgt_e.size(0), tgt_e.size(1), 0).to(device)

            hist_x_original = hist_x.clone()
            hist_e_original = hist_e.clone()

            for i in range(num_samples):
                print("now it is sample:", i)
                p_x = torch.empty(tgt_e.size(0), 0).to(device)
                p_e = torch.empty(tgt_e.size(0), 0).to(device)
                hist_x = hist_x_original.clone()
                hist_e = hist_e_original.clone()
                # for j in range(int(tgt_e.size(1))):
                p_e, p_x = model.sample(hist_x, hist_e, args.tgt_len, history_times)
                pred_x = torch.cat([pred_x, p_x.unsqueeze(-1)], dim=-1)
                pred_e = torch.cat([pred_e, p_e.unsqueeze(-1)], dim=-1)

            if args.boxcox:
                # https://stats.stackexchange.com/questions/541748/simple-problem-with-box-cox-transformation-in-a-time-series-model
                # Why need clamp, this website gives the answer

                pred_x = pred_x * args.train_bc_std + args.train_bc_mean
                if args.train_lambda_boxcox > 0:
                    pred_x[
                        pred_x < -1 / args.train_lambda_boxcox] = -1 / args.train_lambda_boxcox + Constants.EPS * 1000
                else:
                    pred_x[
                        pred_x > -1 / args.train_lambda_boxcox] = -1 / args.train_lambda_boxcox - Constants.EPS * 1000
                pred_x = inv_boxcox(pred_x.cpu(), args.train_lambda_boxcox) / args.scale
                pred_x[pred_x < 0] = ((args.min_inter_time + Constants.EPS) * 0.85).to(args.device)
            else:
                pred_x = pred_x * args.train_ln_std + args.train_ln_mean
                pred_x = torch.exp(pred_x)
                if args.dataset == 'retweet':
                    pred_x = pred_x / Constants.SCALE_RETWEET
                else:
                    pred_x = pred_x / Constants.SCALE_UNIFORM
                pred_x[pred_x < 0] = ((args.min_inter_time + Constants.EPS) * 0.85).to(args.device)

            pred_x[pred_x < 0] = (min_inter_time + Constants.EPS).to(args.device)

            pred_x_total = torch.cat([pred_x_total, pred_x.cpu()], dim=0)
            pred_e_total = torch.cat([pred_e_total, pred_e.cpu()], dim=0)
            gt_e_total = torch.cat([gt_e_total, tgt_e.cpu()], dim=0)
            gt_x_total = torch.cat([gt_x_total, unnormed_target_dt.cpu()], dim=0)

    ###################################################################################################
    ########################################### Record time ###########################################
    ###################################################################################################

    total_sampling_time = time.time() - since

    pred_e_copy = pred_e_total.detach().clone()

    pred_e = pred_e_total.cpu().long()
    pred_x = pred_x_total.cpu()

    gt_e = gt_e_total.cpu().long()
    gt_x = gt_x_total.cpu() + Constants.EPS


    ######################################################################################################
    ############################################ Save Samples ############################################
    ######################################################################################################

    torch.save(pred_x, path_samples_dt)
    torch.save(pred_e_copy.cpu(), path_samples_type)

    ###########################################################################################################
    ############################################ Save Ground Truth ############################################
    ###########################################################################################################

    torch.save(gt_x, path_gt_dt)
    torch.save(gt_e, path_gt_type)

    ######################################################################################################
    ############################################ Take Average ############################################
    ######################################################################################################
    # pred_x_clone = pred_x.detach().clone()[pred_x<gt_x.max()+1]
    pred_x_clone = pred_x.detach().clone()
    pred_e_clone = pred_e.detach().clone()
    pred_x = pred_x.mean(dim=-1).squeeze(-1)
    pred_e = torch.mode(pred_e, dim=-1).values.long()
    gt_e = gt_e
    gt_x = gt_x

    ########################################################################################################
    ############################################ OTD w/o filter ############################################
    ########################################################################################################

    filter = False
    distances_wo_filter = get_distances_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter,
                                                  args.time_range, distance_del_cost, trans_cost)

    ##############################################################################################################
    ############################################ Type RMSE w/o filter ############################################
    ##############################################################################################################

    filter = False
    rmse_types_wo_filter = type_rmse_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter, args.time_range)

    #########################################################################################################
    ############################################ OTD with filter ############################################
    #########################################################################################################

    filter = True
    distances_with_filter = get_distances_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter,
                                                    args.time_range, distance_del_cost, trans_cost)

    ###############################################################################################################
    ############################################ Type RMSE with filter ############################################
    ###############################################################################################################

    filter = True
    rmse_types_with_filter = type_rmse_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter, args.time_range)

    ##################################################################################################################
    ############################################ rmse and mae # of Events ############################################
    ##################################################################################################################

    rmse_num_events, mae_num_events = rmse_mae_num_events_diffusion(pred_x, pred_e, gt_x, gt_e, args.time_range)

    ###################################################################################################
    ############################################ Time RMSE ############################################
    ###################################################################################################

    rmse_mean, rmse_std = time_rmse_tensor(pred_x.cpu(), gt_x.cpu())

    ##############################################################################################
    ############################################ MAPE ############################################
    ##############################################################################################

    mape_mean, mape_std = mape_tensor(pred_x.cpu(), gt_x.cpu())

    ##############################################################################################
    ############################################ sMAPE ###########################################
    ##############################################################################################

    smape_mean, smape_std = sMape_tensor(pred_x.cpu(), gt_x.cpu())

    ###############################################################################################
    ############################################# Log #############################################
    ###############################################################################################

    distances_wo_filter = np.array(distances_wo_filter)
    print('distance (fixed forecasting) mean is {:.3f}'.format(
        distances_wo_filter.mean())
    )

    rmse_types_wo_filter = np.array(rmse_types_wo_filter)
    print('rmse type (fixed forecasting) mean is {:.3f}'.format(
        rmse_types_wo_filter.mean())
    )

    distances_with_filter = np.array(distances_with_filter)
    print('distance (interval forecasting) is {:.3f}'.format(
        distances_with_filter.mean())
    )

    rmse_types_with_filter = np.array(rmse_types_with_filter)
    print('rmse type (interval forecasting) mean is {:.3f}'.format(
        rmse_types_with_filter.mean())
    )

    print('rmse # of events is {: .3f}'.format(rmse_num_events))
    print('mae # of events is {: .3f}'.format(mae_num_events))

    print('rmse time is {:.3f}'.format(rmse_mean))

    print('total sampling time is {total_time: .3f}'.format(total_time=total_sampling_time))
    print('Number of total samples: {}'.format(pred_e_copy.flatten().size(0)))
    print('Number of samples per sequence: {}'.format(num_samples))

    with open(path_samples_result, 'w') as f:
        f.write('distance (fixed forecasting): {:.3f}\n'.format(
            distances_wo_filter.mean())
        )

        f.write('rmse type (fixed forecasting): {:.3f}\n'.format(
            rmse_types_wo_filter.mean())
        )

        f.write('distance (interval forecasting): {:.3f}\n'.format(
            distances_with_filter.mean())
        )

        f.write('rmse type (interval forecasting): {:.3f}\n'.format(
            rmse_types_with_filter.mean())
        )

        f.write('rmse # of events: {: .3f}\n'.format(rmse_num_events))
        f.write('mae # of events: {: .3f}\n'.format(mae_num_events))

        f.write('rmse time: {:.3f}\n'.format(rmse_mean))

        f.write('total sampling time: {total_time: .3f}s\n'.format(total_time=total_sampling_time))
        f.write('Number of total samples: {}\n'.format(pred_e_copy.flatten().size(0)))
        f.write('Number of samples per sequence: {}\n'.format(num_samples))
        f.write('Num of training parameters: {}\n'.format(total_trainable_params))

    save_args(args)

    return args