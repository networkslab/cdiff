import torch.nn as nn
from tpp_utils_seq2seq.layers.history_encoder import HistoryEncoder
from tpp_utils_seq2seq.layers.type_denoising_module import TypeDenoisingModule
from cross_diffusion_utils.type_diffusion_model import DiffusionTypeModel
from cross_diffusion_utils.time_diffusion_model import DiffusionTimeModel
from cross_diffusion_utils.tabular_diffusion_model import DiffusionTabularModel
from tpp_utils_seq2seq.layers.time_denoising_module import TimeDenoisingModule


def add_model_args(parser):
    # Flow params
    parser.add_argument('--diffusion_steps', type=int, default=200)
    # Transformer params.
    parser.add_argument('--transformer_dim', type=int, default=32)
    parser.add_argument('--transformer_heads', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=64)
    parser.add_argument('--num_decoder_layers', type=int, default=1)


class TabDiffEventSeqModel(nn.Module):
    def __init__(self, args, num_classes):
        super(TabDiffEventSeqModel, self).__init__()
        # get hyper-parameters
        num_classes = num_classes
        transformer_dim = args.transformer_dim
        transformer_heads = args.transformer_heads
        diffusion_steps = args.diffusion_steps
        num_encoder_layers = args.num_encoder_layers
        dim_feedforward = args.dim_feedforward
        num_decoder_layers = args.num_decoder_layers
        batch_size = args.batch_size
        tgt_len = args.tgt_len
        device = args.device
        num_timesteps = args.diffusion_steps

        self.device = device
        self.num_classes = num_classes

        self.hist_enc = HistoryEncoder(
            transformer_dim=transformer_dim, transformer_heads=transformer_heads, num_classes=num_classes,
            device=device, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
        )

        self.denoise_fn_type = TypeDenoisingModule(
            transformer_dim=transformer_dim, num_classes=num_classes, n_steps=num_timesteps,
            transformer_heads=transformer_heads, dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers, device=device)

        self.denoise_fn_dt = TimeDenoisingModule(
            transformer_dim=transformer_dim, num_classes=num_classes, n_steps=num_timesteps,
            transformer_heads=transformer_heads, dim_feedforward=dim_feedforward,
            n_decoder_layers=num_decoder_layers, device=device
        )

        self.diffusion_type_model = DiffusionTypeModel(n_steps=args.diffusion_steps, denoise_fn=self.denoise_fn_type,
                                                       num_classes=num_classes)

        self.diffusion_time_model = DiffusionTimeModel(n_steps=args.diffusion_steps, denoise_func=self.denoise_fn_dt)

        self.diffusion_tabular_model = DiffusionTabularModel(hist_enc_func=self.hist_enc,
                                                             time_diffusion_model=self.diffusion_time_model,
                                                             type_diffusion_model=self.diffusion_type_model,
                                                             n_steps=args.diffusion_steps,
                                                             device=args.device,
                                                             num_classes=args.num_classes)

    def forward(self, hist_x, hist_e, tgt_x, tgt_e, hist_time_stamps):
        loss, dt_loss, type_loss = self.diffusion_tabular_model.compute_loss(tgt_x, tgt_e, hist_x, hist_e,
                                                                             hist_time_stamps)
        loss = loss.sum(-1).mean()
        dt_loss = dt_loss.sum(-1).mean()
        type_loss = type_loss.sum(-1).mean()
        return loss, dt_loss, type_loss

    def sample(self, hist_x, hist_e, tgt_len, history_times):
        e, x = self.diffusion_tabular_model.sample_chain(hist_x, hist_e, history_times, tgt_len)
        return log_onehot_to_index(e[-1]), x[-1]

    def sample_ddim(self, hist_x, hist_e, tgt_len, history_times):
        e, x = self.diffusion_tabular_model.sample_chain_ddim(hist_x, hist_e, history_times, tgt_len)
        return log_onehot_to_index(e[-1]), x[-1]

    def sample_chain(self, hist_x, hist_e, tgt_len, history_times):
        e, x = self.diffusion_tabular_model.sample_chain(hist_x, hist_e, history_times, tgt_len)
        e_list = []
        for item in e:
            e_list.append(log_onehot_to_index(item))
        return e_list, x


def log_onehot_to_index(log_e):
    return log_e.argmax(1)


def get_model(args, num_classes):
    return TabDiffEventSeqModel(args, num_classes)


def get_model_id(args):
    if args.boxcox:
        return 'cross_diffusion_discrete_boxcox_{}_tgt_len_{}'.format(args.diffusion_steps, args.tgt_len)
    else:
        return 'cross_diffusion_discrete_ln_{}_tgt_len_{}'.format(args.diffusion_steps, args.tgt_len)