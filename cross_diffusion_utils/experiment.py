from abc import ABC

import torch
from cross_diffusion_utils.utils import get_args_table, clean_dict
import torch.nn as nn
# Path
import os
import time
import pathlib
HOME = str(pathlib.Path.home())

# Experiment
from cross_diffusion_utils.base import BaseExperiment
from cross_diffusion_utils.base import DataParallelDistribution

#  Logging frameworks
# from torch.utils.tensorboard import SummaryWriter
# import mlflow


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--need_regularization', action='store_true')

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)

    # Eval params
    parser.add_argument('--eval_num_samples', type=int, default=5)
    parser.add_argument('--distance_del_cost', type=list, default=[0.05, 0.5, 1, 1.5, 2, 3, 4])
    parser.add_argument('--trans_cost', type=float, default=1.0)


class DiffusionExperiment(BaseExperiment, ABC):
    no_log_keys = ['project', 'name',
                   'log_tb',
                   'check_every', 'eval_every',
                   'device', 'parallel'
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):
        if args.log_home is None:
            self.log_base = os.path.join(HOME, 'log', 'flow')
        else:
            self.log_base = os.path.join(args.log_home, 'log', 'flow')

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.eval_every
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        args.log_path = os.path.join(self.log_base, data_id, model_id, optim_id, args.name)

        # Move model
        model = model
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)

        # Create EMA model
        ema = EMA(0.9)
        ema.register(model)

        # Init parent
        super(DiffusionExperiment, self).__init__(model=model,ema=ema,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, data_id, model_id, optim_id, args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every)
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        # if args.log_tb:
        #     mlflow.set_tracking_uri(uri=self.log_base)
        #     mlflow.set_experiment(data_id+'_'+model_id+'_'+optim_id)
        #     mlflow.start_run(run_name=data_id+'_'+model_id+'_'+optim_id)


    def log_fn(self, epoch, train_dict, eval_dict):
        pass
        # Tensorboard
        # if self.args.log_tb:
        #     for metric_name, metric_value in train_dict.items():
        #         mlflow.log_metrics({'base/{}'.format(metric_name): metric_value}, step=epoch+1)
        #     mlflow.log_metrics({'base/lr': self.optimizer.param_groups[0]['lr']}, step=epoch + 1)
        #     if eval_dict:
        #         for metric_name, metric_value in eval_dict.items():
        #             mlflow.log_metrics({'eval/{}'.format(metric_name): metric_value}, step=epoch+1)

    def resume(self):
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: eval_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict)

    def run(self):
        if self.args.resume: self.resume()
        super(DiffusionExperiment, self).run(epochs=self.args.epochs)


