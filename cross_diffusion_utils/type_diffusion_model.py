import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
import math

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    x[x == num_classes] = num_classes-1
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


class DiffusionTypeModel(torch.nn.Module):
    def __init__(self, num_classes, denoise_fn, n_steps=100,
                 loss_type='vb_stochastic', parametrization='x0', eta=1):
        super(DiffusionTypeModel, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        # self.shape = shape
        self.n_steps = n_steps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(n_steps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_prev_cumprod_alpha = torch.cat([torch.Tensor([1]), log_cumprod_alpha[:-1]])

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # sigma = log_prev_cumprod_alpha
        # eta = 1
        # self.eta = eta

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        # self.register_buffer('log_prev_cumprod_alpha', log_prev_cumprod_alpha.float())
        self.register_buffer('Lt_history', torch.zeros(n_steps))
        self.register_buffer('Lt_count', torch.zeros(n_steps))
        # self.register_buffer('sigma', sigma.float())

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, log_x_t, t, dt, hist, non_padding_mask):
        x_t = log_onehot_to_index(log_x_t)
        # t, e, x, hist, non_padding_mask
        out = self._denoise_fn(t, x_t, dt, hist, non_padding_mask)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred



    def q_posterior_ddim(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)





    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, dt, hist, non_padding_mask):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)
        else:
            raise ValueError
        return log_model_pred


    def p_pred_ddim(self, log_x, t, dt, hist, non_padding_mask):

        log_x_recon = self.predict_start(log_x, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)
        log_model_pred = self.q_posterior(
            log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x_recon

    def p_pred_accelerate(self, log_x, log_x_recon, t):
        log_model_pred = self.q_posterior(
            log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x_recon
    @torch.no_grad()
    def p_sample(self, log_x, t, dt, hist, non_padding_mask):
        model_log_prob = self.p_pred(log_x=log_x, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)
        out = self.log_sample_categorical(model_log_prob)
        return out


    @torch.no_grad()
    def p_sample_ddim(self, log_x, t, dt, hist, non_padding_mask):
        model_log_prob, log_x_recon= self.p_pred_ddim(log_x=log_x, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)
        out = self.log_sample_categorical(model_log_prob)
        return out, log_x_recon

    @torch.no_grad()
    def p_sample_accelerate(self,log_x, log_x_recon, t):
        model_log_prob, log_x_recon = self.p_pred_accelerate(log_x, log_x_recon, t)
        out = self.log_sample_categorical(model_log_prob)
        return out, log_x_recon

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.n_steps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.n_steps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        # return sum_except_batch(kl_prior)
        return kl_prior

    def compute_Lt(self, log_x_start, log_x_t, t, dt, hist, non_padding_mask, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t, dt=dt, hist=hist, non_padding_mask=non_padding_mask)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        # kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        # decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        mask = mask.repeat(decoder_nll.size(1),1)
        mask = mask.permute(1,0)
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.n_steps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.n_steps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, dt, hist, non_padding_mask, t, pt):
        b, device = x.size(0), x.device

        if self.loss_type == 'vb_stochastic':
            x_start = x

            log_x_start = index_to_log_onehot(x_start, self.num_classes)

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, dt, hist, non_padding_mask)

            Lt2 = kl.pow(2).sum(-1)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt.unsqueeze(-1) + kl_prior
            return -vb_loss

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(x)
        else:
            raise ValueError()

    def compute_loss(self, x, dt, hist, non_padding_mask, t, pt):
        # x: e, dt: x, hist: hist,
        num_elem = x.size(0)
        loss = self._train_loss(x, dt, hist, non_padding_mask, t, pt)
        loss = -loss / (math.log(2))
        return loss

    def log_prob(self, x, dt, hist, non_padding_mask, t, pt):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x, dt, hist, non_padding_mask, t, pt)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)
            #
            # t, pt = self.sample_time(b, device, 'uniform')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, dt, hist, non_padding_mask)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior
            num_elem = x.flatten().size(0)
            loss = loss.sum()/(math.log(2) * num_elem)

            return -loss

    def sample(self, num_samples, dt, hist, non_padding_mask):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.n_steps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, dt, hist, non_padding_mask)
        print()
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.n_steps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.n_steps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, dt, hist, non_padding_mask)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs