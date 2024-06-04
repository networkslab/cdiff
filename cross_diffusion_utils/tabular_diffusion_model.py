import torch
import torch.nn as nn
import numpy as np
from cross_diffusion_utils.time_diffusion_model import DiffusionTimeModel
from cross_diffusion_utils.type_diffusion_model import DiffusionTypeModel
import torch.nn.functional as F


class DiffusionTabularModel(torch.nn.Module):
    def __init__(self, hist_enc_func, time_diffusion_model, type_diffusion_model, n_steps=100, device='cuda',
                 num_classes=5):
        super(DiffusionTabularModel, self).__init__()
        self.n_steps = n_steps
        self.type_diff_ = type_diffusion_model
        self.time_diff_ = time_diffusion_model
        self.hist_enc_func_ = hist_enc_func
        self.device = device
        self.num_classes = num_classes

    def compute_loss(self, tgt_x, tgt_e, hist_x, hist_e, hist_time):
        hist, non_padding_mask = self.get_hist(hist_x, hist_e, hist_time)
        t, pt = self.sample_time(tgt_e.size(0), device=self.device)
        type_loss = self.type_diff_.compute_loss(tgt_e, tgt_x, hist, non_padding_mask, t, pt)
        time_loss = self.time_diff_.compute_loss(tgt_x, tgt_e, hist, non_padding_mask, t)
        return time_loss + type_loss, time_loss, type_loss

    def get_hist(self, hist_x, hist_e, hist_time):
        non_pad_mask = self.get_non_pad_mask(hist_e)
        hist = self.hist_enc_func_(hist_x=hist_x, hist_e=hist_e, hist_time_stamps=hist_time, non_pad_mask=non_pad_mask)
        return hist, non_pad_mask

    def sample_time(self, b, device, method='uniform'):
        t = torch.randint(0, self.n_steps, (b,), device=device).long()

        pt = torch.ones_like(t).float() / self.n_steps
        return t, pt

    def get_non_pad_mask(self, seq):
        """
        Get the non-padding positions.
        Set num_classes as padding, then 0 is an actual event type.
        seq: should be history event sequence, the target sequence is fixed length, default 20, B x L'
        output: B x L'
        """

        assert seq.dim() == 2
        return seq.eq(self.num_classes)

    def sample_chain(self, hist_x, hist_e, hist_time, tgt_len):
        hist, non_padding_mask = self.get_hist(hist_x, hist_e, hist_time)

        shape = [hist.size(0), tgt_len]
        init_x = torch.randn(shape).to(self.device)
        # x_t_list = [init_x.unsqueeze(0)]
        x_t_list = [init_x]

        x_t = init_x

        shape = (tgt_len,)
        b = hist.size(0)
        uniform_logits = torch.zeros(
            (b, self.num_classes,) + shape, device=self.device)
        # zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        e_t = self.log_sample_categorical(uniform_logits)

        e_t_list = [e_t]

        for i in reversed(range(0, self.n_steps)):
            e_t_index = log_onehot_to_index(e_t)
            x_seq = self.time_diff_._one_diffusion_rev_step(self.time_diff_.denoise_func_, x_t, e_t_index, i, hist,
                                                            non_padding_mask)
            x_t_list.append(x_seq)
            # e_t, t, x_t, hist, non_padding_mask
            t_type = torch.full((b,), i, device=self.device, dtype=torch.long)
            e_seq = self.type_diff_.p_sample(e_t, t_type, x_t, hist, non_padding_mask)
            e_t_list.append(e_seq)
            x_t = x_seq
            e_t = e_seq
        return e_t_list, x_t_list

    def sample_chain_ddim(self, hist_x, hist_e, hist_time, tgt_len):
        hist, non_padding_mask = self.get_hist(hist_x, hist_e, hist_time)

        shape = [hist.size(0), tgt_len]
        init_x = torch.randn(shape).to(self.device)
        # x_t_list = [init_x.unsqueeze(0)]
        x_t_list = [init_x]

        x_t = init_x

        shape = (tgt_len,)
        b = hist.size(0)
        uniform_logits = torch.zeros(
            (b, self.num_classes,) + shape, device=self.device)
        # zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        e_t = self.log_sample_categorical(uniform_logits)

        # e_t_list = [e_t.unsqueeze(0)]
        e_t_list = [e_t]

        diffusion_process = self.time_diff_._get_process_scheduling()
        # for step in reversed(range(i, prev_i)):
        #     e_t =
        for prev_i, i in diffusion_process:

            e_t_index = log_onehot_to_index(e_t)

            t_type = torch.full((b,), i, device=self.device, dtype=torch.long)
            e_seq, e_0_pred = self.type_diff_.p_sample_ddim(e_t, t_type, x_t, hist, non_padding_mask)
            e_t_list.append(e_seq)

            x_seq = self.time_diff_._one_diffusion_rev_step_ddim(self.time_diff_.denoise_func_, x_t, e_t_index, i, hist,
                                                                 non_padding_mask, prev_i)
            x_t_list.append(x_seq)
            # for step in reversed(range(i, prev_i)):

            for step in range(prev_i, i):
                t_type = torch.full((b,), step, device=self.device, dtype=torch.long)
                e_seq, e_0_pred = self.type_diff_.p_sample_accelerate(e_seq, e_0_pred, t_type)

            x_t = x_seq
            e_t = e_seq
        return e_t_list, x_t_list

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_e):
    return log_e.argmax(1)
