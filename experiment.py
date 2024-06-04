import math
import torch
from cross_diffusion_utils.experiment import DiffusionExperiment, add_exp_args
import numpy as np
# Metric
from metrics import get_distances_diffusion, time_rmse_tensor, type_rmse_diffusion, sMape_metric
from scipy.special import boxcox, inv_boxcox

import tpp_utils_seq2seq.dataset_seq2seq.Constants as Constants


class Experiment(DiffusionExperiment):
    def train_fn(self, epoch):

        self.model.train()
        loss_sum = 0.0
        dt_loss_sum = 0.0
        type_loss_sum = 0.0
        loss_count = 0
        loss_moving = None

        for iteration, batch in enumerate(self.train_loader):

            history_times = batch.history_times
            hist_e = batch.history_types.long()
            hist_x = batch.history_dt
            tgt_e = batch.target_types.long()
            tgt_x = batch.target_dt
            loss, dt_loss, type_loss = self.model(hist_x, hist_e, tgt_x, tgt_e, history_times)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

            self.ema.update(self.model)

            if (iteration + 1) % self.args.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # if self.scheduler_iter: self.scheduler_iter.step()
                self.scheduler_epoch.step()

            loss_sum += loss.detach().cpu().item() * len(tgt_x)
            dt_loss_sum += dt_loss.detach().cpu().item() * len(tgt_x)
            type_loss_sum += type_loss.detach().cpu().item() * len(tgt_x)
            loss_count += len(tgt_x)

            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            if self.args.debug and loss_count > self.args.debug:
                break
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Total loss : {:.3f}, '
                  'Total time loss: {:.3f}, Total type loss: {:.3f}'.format(epoch + 1, self.args.epochs,
                                                                            loss_count,
                                                                            len(self.train_loader.dataset),
                                                                            loss_sum / loss_count,
                                                                            dt_loss_sum / loss_count,
                                                                            type_loss_sum / loss_count), end='\r')
        print('')
        # if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'total_loss': loss_sum / loss_count, 'dt_loss': dt_loss_sum / loss_count,
                'type_loss': type_loss_sum / loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        print()
        with torch.no_grad():
            loss_sum = 0.0
            dt_loss_sum = 0.0
            type_loss_sum = 0.0
            loss_count = 0

            batch_count = 0
            total_distances_wo_filter = []
            total_rmse_types_wo_filter = []
            total_smape = []

            for iteration, batch in enumerate(self.eval_loader):
                history_times = batch.history_times
                hist_e = batch.history_types.long()
                hist_x = batch.history_dt
                tgt_e = batch.target_types.long()
                tgt_x = batch.target_dt
                unnormed_target_dt = batch.unnormed_target_dt

                batch_count += 1

                pred_e, pred_x = self.model.sample(hist_x, hist_e, self.args.tgt_len, history_times)

                if self.args.boxcox:
                    # https://stats.stackexchange.com/questions/541748/simple-problem-with-box-cox-transformation-in-a-time-series-model
                    # Why need clamp, this website gives the answer

                    pred_x = pred_x * self.args.train_bc_std + self.args.train_bc_mean
                    if self.args.train_lambda_boxcox > 0:
                        pred_x[
                            pred_x < -1 / self.args.train_lambda_boxcox] = -1 / self.args.train_lambda_boxcox + Constants.EPS * 100
                    else:
                        pred_x[
                            pred_x > -1 / self.args.train_lambda_boxcox] = -1 / self.args.train_lambda_boxcox - Constants.EPS * 100
                    pred_x = inv_boxcox(pred_x.cpu(), self.args.train_lambda_boxcox) / self.args.scale
                    pred_x[pred_x < 0] = ((self.args.min_inter_time + Constants.EPS) * 0.85).to(self.args.device)
                else:
                    pred_x = pred_x * self.args.train_ln_std + self.args.train_ln_mean
                    pred_x = torch.exp(pred_x)
                    if self.args.dataset == 'retweet':
                        pred_x = pred_x / Constants.SCALE_RETWEET
                    else:
                        pred_x = pred_x / Constants.SCALE_UNIFORM
                    pred_x[pred_x < 0] = ((self.args.min_inter_time + Constants.EPS) * 0.85).to(self.args.device)

                pred_x[pred_x < 0] = (self.args.min_inter_time * 0.85 + Constants.EPS).to(self.args.device)

                pred_e = pred_e.cpu().long()
                pred_x = pred_x.cpu()
                gt_e = tgt_e.cpu().long()
                gt_x = unnormed_target_dt.cpu() + Constants.EPS
                gt_e = gt_e
                gt_x = gt_x

                ########################################################################################################
                ############################################ OTD w/o filter ############################################
                ########################################################################################################

                filter = False
                distances_wo_filter = get_distances_diffusion(pred_x, pred_e, gt_x, gt_e, self.args.num_classes, filter,
                                                              self.args.time_range, self.args.distance_del_cost,
                                                              self.args.trans_cost)

                total_distances_wo_filter += list(np.array(distances_wo_filter))

                ##############################################################################################################
                ############################################ Type RMSE w/o filter ############################################
                ##############################################################################################################

                filter = False
                rmse_types_wo_filter = type_rmse_diffusion(pred_x, pred_e, gt_x, gt_e, self.args.num_classes, filter,
                                                           self.args.time_range)

                total_rmse_types_wo_filter += list(np.array(rmse_types_wo_filter))

                ##############################################################################################
                ############################################ sMAPE ###########################################
                ##############################################################################################

                s_ape = sMape_metric(pred_x.cpu(), gt_x.cpu())
                total_smape += list(np.array(s_ape))

                loss, dt_loss, type_loss = self.model(hist_x, hist_e, tgt_x, tgt_e, history_times)
                loss_sum += loss.detach().cpu().item() * len(tgt_e)
                dt_loss_sum += dt_loss.detach().cpu().item() * len(tgt_e)
                type_loss_sum += type_loss.detach().cpu().item() * len(tgt_e)
                loss_count += len(tgt_e)

            # Find mean to report for val
            total_distances_wo_filter = np.mean(total_distances_wo_filter)
            total_rmse_types_wo_filter = np.mean(total_rmse_types_wo_filter)
            total_smape = np.mean(total_smape)
            print('Evaluating train for N={} forecasting. Epoch: {}/{}, Datapoint: {}/{}, Total loss: {:.3f}, '
                  'Time loss: {:.3f}, Type loss: {:3f} '
                  'OTD fixed forecasting: {:.3f}, rmse_type fixed forecasting: {:.3f},'
                  'sMAPE: {:.3f}'.format(self.args.tgt_len,
                epoch + 1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum / loss_count,
                dt_loss_sum / loss_count, type_loss_sum / loss_count,
                total_distances_wo_filter, total_rmse_types_wo_filter, total_smape), end='\r')
            print('')

        return {'total_loss': loss_sum / loss_count, 'dt_loss': dt_loss_sum / loss_count,
                'type_loss': type_loss_sum / loss_count,
                'otd_wo_filter': total_distances_wo_filter,
                'rmse_type_wo_filter': total_rmse_types_wo_filter,
                'smape': total_smape}
