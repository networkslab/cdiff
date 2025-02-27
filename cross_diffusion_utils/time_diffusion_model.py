import torch
import torch.nn as nn
import numpy as np


class DiffusionTimeModel(nn.Module):
    def __init__(self, denoise_func, n_steps=100, device='cuda', tau=50):
        super().__init__()
        # self.cur_x = torch.randn([5000, 20]).to(device)
        self.n_steps = n_steps

        betas = self.make_beta_schedule(schedule='linear', n_timesteps=n_steps, start=1e-6, end=1e-2).to(device)
        self.register_buffer('betas', betas)
        alphas = 1 - betas
        self.register_buffer('alphas', alphas)
        alphas_prod = torch.cumprod(alphas, 0).to(device)
        alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0).to(device)
        self.register_buffer('alpha_bars', alphas_prod)
        self.register_buffer('alpha_prev_bars', alphas_prod_p)
        alphas_prod_p_sqrt = alphas_prod_p.sqrt().to(device)
        self.register_buffer('alphas_prod_p_sqrt', alphas_prod_p_sqrt)
        # self.alphas_prod_p_sqrt = alphas_prod_p_sqrt
        alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
        self.register_buffer('alphas_bar_sqrt', alphas_bar_sqrt)
        # self.alphas_bar_sqrt = alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)
        self.register_buffer('one_minus_alphas_bar_sqrt', one_minus_alphas_bar_sqrt)
        # self.one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)
        sqrt_recip_alphas_cumprod = (1 / alphas_prod).sqrt().to(device)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.to(device)
        sqrt_alphas_cumprod_m1 = (1 - alphas_prod).sqrt() * sqrt_recip_alphas_cumprod.to(device)
        self.register_buffer('sqrt_alphas_cumprod_m1', sqrt_alphas_cumprod_m1)
        # self.sqrt_alphas_cumprod_m1 = sqrt_alphas_cumprod_m1.to(device)
        posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod)).to(device)
        self.register_buffer('posterior_mean_coef_1', posterior_mean_coef_1)
        # self.posterior_mean_coef_1 = posterior_mean_coef_1.to(device)
        posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod)).to(device)
        self.register_buffer('posterior_mean_coef_2', posterior_mean_coef_2)
        # self.posterior_mean_coef_2 = posterior_mean_coef_2.to(device)
        posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
        self.register_buffer('posterior_variance', posterior_variance)
        posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        # self.posterior_log_variance_clipped = posterior_log_variance_clipped.to(device)
        self.denoise_func_ = denoise_func
        self.device = device
        self.tau = tau


    def make_beta_schedule(sefl, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_t = self.extract(self.alphas_bar_sqrt, t.long(), x_0)
        alphas_1_m_t = self.extract(self.one_minus_alphas_bar_sqrt, t.long(), x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    def sample_continuous_noise_level(self, batch_size):
        """
        Samples continuous noise level.
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """

        t = np.random.choice(range(1, self.n_steps), size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.alphas_prod_p_sqrt[t - 1].cpu(),
                self.alphas_prod_p_sqrt[t].cpu(),
                size=batch_size
            ))

        return continuous_sqrt_alpha_cumprod.unsqueeze(-1).to(self.device)

    # def q_sample(self, x_0, continuous_sqrt_alpha_cumprod=None, eps=None):
    #     batch_size = x_0.shape[0]
    #     if isinstance(eps, type(None)):
    #         continuous_sqrt_alpha_cumprod = self.sample_continuous_noise_level(batch_size)
    #         eps = torch.randn_like(x_0)
    #     # Closed form signal diffusion
    #     outputs = continuous_sqrt_alpha_cumprod * x_0 + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * eps
    #     return outputs

    def q_posterior(self, x_start, x, t):
        """ Computes reverse (denoising) process posterior q(y_{t-1}|y_0, y_t, x) """
        posterior_mean = self.posterior_mean_coef_1[t] * x_start + self.posterior_mean_coef_2[t] * x
        posterior_log_variance_clip = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clip

    def predict_start_from_noise(self, x, t, eps):
        """ Computes y_0 from given y_t and reconstructed noise. """
        return self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_alphas_cumprod_m1[t] * eps

    def p_mean_variance(self, model, x, e, t, hist, non_padding_mask, clip_denoised=True):
        """ Computes Gaussian transitions of Markov chain at step t """
        batch_size = x.shape[0]
        # noise_level = torch.FloatTensor([self.alphas_prod_p_sqrt[t + 1]]).repeat(batch_size, 1).to(self.device)
        # noise_level = torch.FloatTensor([self.alphas_prod_p_sqrt[t + 1]]).repeat(batch_size, 1).to(self.device)
        noise_level = torch.FloatTensor([t]).repeat(batch_size, 1).to(self.device)
        # Infer noise, conditioned on continuous level
        eps_recon = model(x.type(torch.cuda.FloatTensor), e, noise_level.type(torch.cuda.FloatTensor), hist, non_padding_mask)
        eps_recon = eps_recon.squeeze(-1)
        x_recon = self.predict_start_from_noise(x, t, eps_recon)
        # Output clipping in WaveGrad
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, model, x, e, t, hist, non_padding_mask):
        model_mean, model_log_variance = self.p_mean_variance(model, x.to(self.device), e, t, hist, non_padding_mask)
        eps = torch.randn_like(x).to(self.device) if t > 0 else torch.zeros_like(x)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    def _one_diffusion_rev_step(self, model, cur_x, e, i, hist, non_padding_mask):
        noise = torch.zeros_like(cur_x) if i == 0 else torch.randn_like(cur_x)
        sqrt_tilde_beta = torch.sqrt((1-self.alpha_prev_bars[i])/(1-self.alpha_bars[i])*self.betas[i])
        noise_level = torch.FloatTensor([i]).repeat(cur_x.size(0), 1).to(self.device)
        pred_eps = self.denoise_func_(cur_x.type(torch.cuda.FloatTensor), e, noise_level.type(torch.cuda.FloatTensor), hist, non_padding_mask)

        mu_theta_xt = torch.sqrt(1/self.alphas[i])*(cur_x-self.betas[i]/(torch.sqrt(1-self.alpha_bars[i]))*pred_eps.squeeze(-1))
        x = mu_theta_xt + sqrt_tilde_beta*noise
        return x

    def _get_process_scheduling(self, reverse=True):
        diffusion_process = (np.linspace(0, np.sqrt(self.n_steps*0.8), 20)**2)
        diffusion_process = [int(s) for s in list(diffusion_process)] + [self.n_steps-1]
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:]) if reverse else
                                zip(diffusion_process[1:], diffusion_process[:-1]))

        return diffusion_process

    def _one_diffusion_rev_step_ddim(self, model, cur_x, e, i, hist, non_padding_mask, prev_i):
        noise_level = torch.FloatTensor([i]).repeat(cur_x.size(0), 1).to(self.device)
        pred_eps = self.denoise_func_(cur_x.type(torch.cuda.FloatTensor), e, noise_level.type(torch.cuda.FloatTensor),
                                      hist, non_padding_mask).squeeze(-1)
        # pred_x_0 = torch.sqrt(self.alpha_bars[prev_i] * (cur_x - torch.sqrt(1-self.alpha_bars[i]) * pred_eps) / torch.sqrt(self.alpha_bars[i]))
        pred_x_0 = torch.sqrt(self.alpha_bars[prev_i]) * (cur_x - self.one_minus_alphas_bar_sqrt[i] * pred_eps) / torch.sqrt(self.alpha_bars[i])
        direction_point_to_xt = torch.sqrt(1-self.alpha_bars[prev_i])*pred_eps
        x = pred_x_0 + direction_point_to_xt
        return x


    def p_sample_loop(self, model, e, hist, non_padding_mask, tgt_len=20):
        shape=[hist.size(0), tgt_len]
        cur_x = torch.randn(shape).to(self.device)
        x_seq = [cur_x.unsqueeze(0)]
        for i in reversed(range(self.n_steps - 1)):
            # cur_x = self.p_sample(model, cur_x, e, i, hist, non_padding_mask)
            cur_x = self._one_diffusion_rev_step(model, cur_x, e, i, hist, non_padding_mask)
            x_seq.append(cur_x.unsqueeze(0))
        return x_seq

    def compute_loss(self, x_0, e, hist, non_padding_mask, t):
        # Sample continuous noise level
        batch_size = x_0.shape[0]
        # t = np.random.choice(range(1, self.n_steps), size=batch_size)
        # continuous_sqrt_alpha_cumprod = self.sample_continuous_noise_level(batch_size)
        t = torch.tensor(t).to(self.device)
        eps = torch.randn_like(x_0)
        # Diffuse the signal
        # y_noisy = self.q_sample(x_0, continuous_sqrt_alpha_cumprod, eps)
        y_noisy = self.q_sample(x_0, t=t, noise=eps)
        # Reconstruct the added noise
        # eps_recon = self.denoise_func_(y_noisy, continuous_sqrt_alpha_cumprod, hist, non_padding_mask)
        eps_recon = self.denoise_func_(y_noisy, e, t, hist, non_padding_mask)
        loss = torch.abs(eps_recon.squeeze(-1)-eps)
        # loss = torch.nn.L1Loss()(eps_recon.squeeze(-1), eps)
        return loss
