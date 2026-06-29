import copy
import math
import torch
import random
import numpy as np
import time as Time
# from thop import profile  # 确保已安装pip install thop
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


# Modified from DiffusionDet and pytorch-diffusion-model
# 中期融合双交叉注意力和改进掩码的结合，并且堆叠了多层交叉注意力

########################################################################################

def get_timestep_embedding(timesteps, embedding_dim):  # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def swish(x):
    return x * torch.sigmoid(x)


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normalize(x, scale):  # [0,1] > [-scale, scale]
    x = (x * 2 - 1.) * scale
    return x


def denormalize(x, scale):  # [-scale, scale] > [0,1]
    x = ((x / scale) + 1) / 2
    return x


######################################################################################

class ASDiffusionModel(nn.Module):
    def __init__(self, encoder_params_rgb, encoder_params_flow, decoder_params, diffusion_params, num_classes, device):
        super(ASDiffusionModel, self).__init__()

        self.device = device
        self.num_classes = num_classes

        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ################################################################

        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']

        self.use_instance_norm = encoder_params_rgb['use_instance_norm']
        if self.use_instance_norm:
            self.ins_norm = nn.InstanceNorm1d(encoder_params_rgb['input_dim'], track_running_stats=False)

        decoder_params['input_dim'] = len(
            [i for i in encoder_params_rgb['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params_rgb['num_f_maps']
        if -1 in encoder_params_rgb['feature_layer_indices']:  # -1 means "video feature"
            decoder_params['input_dim'] += encoder_params['input_dim']
        if -2 in encoder_params_rgb['feature_layer_indices']:  # -2 means "encoder prediction"
            decoder_params['input_dim'] += self.num_classes

        decoder_params['num_classes'] = num_classes
        encoder_params_rgb['num_classes'] = num_classes
        encoder_params_flow['num_classes'] = num_classes
        encoder_params_flow['input_dim'] = 1024
        encoder_params_rgb.pop('use_instance_norm')
        encoder_params_flow.pop('use_instance_norm')

        self.encoder_rgb = EncoderModel(**encoder_params_rgb)
        self.encoder_flow = EncoderModel(**encoder_params_flow)
        self.cross_attn = MultiLayerAttention(2, 192, 64, 0.5, True)#768或者192
        self.decoders = nn.ModuleList([copy.deepcopy(DecoderModel(**decoder_params)) for _ in range(1)])

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):  # forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def model_predictions(self, backbone_feats, x, t):

        x_m = torch.clamp(x, min=-1 * self.scale, max=self.scale)  # [-scale, +scale]
        x_m = denormalize(x_m, self.scale)  # [0, 1]

        assert (x_m.max() <= 1 and x_m.min() >= 0)
        #         flops, _ = profile(self.decoder, inputs=(backbone_feats,t, x_m.float()), verbose=False) #计算flops
        for decoder in self.decoders:
            x_start = decoder(backbone_feats, t, x_m.float())  # torch.Size([1, C, T])
            x_m = x_start
        x_start = F.softmax(x_start, 1)
        assert (x_start.max() <= 1 and x_start.min() >= 0)

        x_start = normalize(x_start, self.scale)  # [-scale, +scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def prepare_targets(self, event_gt):

        # event_gt: normalized [0, 1]

        assert (event_gt.max() <= 1 and event_gt.min() >= 0)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

        noise = torch.randn(size=event_gt.shape, device=self.device)

        x_start = (event_gt * 2. - 1.) * self.scale  # [-scale, +scale]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        event_diffused = ((x / self.scale) + 1) / 2.  # normalized [0, 1]

        return event_diffused, noise, t

    def forward(self, backbone_feats, t, event_diffused, event_gt=None, boundary_gt=None):  # only for train

        if self.detach_decoder:
            backbone_feats = backbone_feats.detach()

        assert (event_diffused.max() <= 1 and event_diffused.min() >= 0)
        cond_type = random.choice(self.cond_types)

        if cond_type == 'full':
            i = 0
            for decoder in self.decoders:
                event_out = decoder(backbone_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'anticipative':
            i = 0
            alpha = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

            batch_size, _, T = backbone_feats.shape
            N_obs = int(T * alpha)
            feature_mask = torch.zeros_like(backbone_feats)
            feature_mask[:, :, :N_obs] = 1.0  # 前N_obs帧可见

            # 迭代解码器
            for decoder in self.decoders:
                masked_feats = feature_mask * backbone_feats  # 应用掩码
                event_out = decoder(masked_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'random':
            i = 0
            w = 10  # 每个片段10帧
            # 生成随机掩码 [batch_size, T]
            batch_size, _, T = backbone_feats.shape
            N_p = (T + w - 1) // w  # 计算总片段数
            N_R = 20 # 需在类中定义self.dataset属性

            # 随机选择N_R个片段掩码
            feature_mask = torch.ones(batch_size, T, device=backbone_feats.device)
            for b in range(batch_size):
                masked_clips = torch.randperm(N_p)[:N_R]  # 随机选片段
                for clip_idx in masked_clips:
                    start = clip_idx * w
                    end = min(start + w, T)
                    feature_mask[b, start:end] = 0.0

            # 调整维度并应用 [batch_size, 1, T]
            feature_mask = feature_mask.unsqueeze(1)

            # 迭代解码器
            for decoder in self.decoders:
                masked_feats = feature_mask * backbone_feats  # 应用掩码
                event_out = decoder(masked_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'zero':
            i = 0
            for decoder in self.decoders:
                event_out = decoder(torch.zeros_like(backbone_feats), t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'boundary05-':
            i = 0
            feature_mask = (boundary_gt < 0.5).float()  # maybe try 0.1
            for decoder in self.decoders:
                event_out = decoder(feature_mask * backbone_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'boundary03-':
            i = 0
            feature_mask = (boundary_gt < 0.3).float()  # maybe try 0.1
            for decoder in self.decoders:
                event_out = decoder(feature_mask * backbone_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                #                 print(event_out[0].shape)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1
        elif cond_type == 'segment=1':
            i = 0
            event_gt = torch.argmax(event_gt, dim=1, keepdim=True).long()  # 1, 1, T
            events = torch.unique(event_gt)
            random_event = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != random_event).float()
            for decoder in self.decoders:
                event_out = decoder(feature_mask * backbone_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1

        elif cond_type == 'segment=2':
            i = 0
            event_gt = torch.argmax(event_gt, dim=1, keepdim=True).long()  # 1, 1, T
            events = torch.unique(event_gt)
            random_event_1 = np.random.choice(events.cpu().numpy())
            random_event_2 = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != random_event_1).float() * (event_gt != random_event_2).float()

            for decoder in self.decoders:
                event_out = decoder(feature_mask * backbone_feats, t, event_diffused.float())
                event_diffused = event_out
                if i == 0:
                    event_outs = event_out.unsqueeze(0)
                event_outs = torch.cat((event_outs, event_out.unsqueeze(0)), dim=0)
                i += 1

        else:
            raise Exception('Invalid Cond Type')
        return event_outs

    def get_training_loss(self, video_feats_rgb, video_feats_flow, event_gt, boundary_gt,proposal_prob,
                          encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
                          decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion,
                          soft_label):

        if self.use_instance_norm:
            video_feats_rgb = self.ins_norm(video_feats_rgb)
            video_feats_flow = self.ins_norm(video_feats_flow)

        encoder_out_rgb, backbone_feats_rgb = self.encoder_rgb(video_feats_rgb, get_features=True)
        encoder_out_flow, backbone_feats_flow = self.encoder_flow(video_feats_flow, get_features=True)
        backbone_feats = self.cross_attn(backbone_feats_rgb, backbone_feats_flow)
        if soft_label is None:
            encoder_ce_loss_rgb = encoder_ce_criterion(
                encoder_out_rgb.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long()  # batch_size must = 1
            )
            encoder_ce_loss_flow = encoder_ce_criterion(
                encoder_out_flow.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long()  # batch_size must = 1
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)

            encoder_ce_loss_rgb = - soft_event_gt * F.log_softmax(encoder_out_rgb, 1)
            encoder_ce_loss_flow = - soft_event_gt * F.log_softmax(encoder_out_flow, 1)
            encoder_ce_loss_rgb = encoder_ce_loss_rgb.sum(0).sum(0)
            encoder_ce_loss_flow = encoder_ce_loss_flow.sum(0).sum(0)

        encoder_mse_loss_rgb = torch.clamp(encoder_mse_criterion(
            F.log_softmax(encoder_out_rgb[:, :, 1:], dim=1),
            F.log_softmax(encoder_out_rgb.detach()[:, :, :-1], dim=1)),
            min=0, max=16)

        encoder_mse_loss_flow = torch.clamp(encoder_mse_criterion(
            F.log_softmax(encoder_out_flow[:, :, 1:], dim=1),
            F.log_softmax(encoder_out_flow.detach()[:, :, :-1], dim=1)),
            min=0, max=16)

        encoder_boundary_loss = torch.tensor(0).to(self.device)  # No boundary loss for encoder
        encoder_ce_loss_rgb = encoder_ce_loss_rgb.mean()
        encoder_ce_loss_flow = encoder_ce_loss_flow.mean()
        encoder_mse_loss_rgb = encoder_mse_loss_rgb.mean()
        encoder_mse_loss_flow = encoder_mse_loss_flow.mean()

        ##########

        event_diffused, noise, t = self.prepare_targets(event_gt)

        # EAST proposal as noisy segmentation prior
        proposal_prob = proposal_prob.to(event_diffused.device).float()

        if proposal_prob.shape[-1] != event_diffused.shape[-1]:
            proposal_prob = F.interpolate(proposal_prob, size=event_diffused.shape[-1], mode='nearest')

        proposal_prob = torch.clamp(proposal_prob, min=1e-8)
        proposal_prob = proposal_prob / proposal_prob.sum(dim=1, keepdim=True)

        # alpha 越大，越接近原始 DiffAct GT 加噪训练；越小，越依赖 EAST proposal
        alpha = 0.5
        event_diffused = alpha * event_diffused + (1 - alpha) * proposal_prob
        event_diffused = torch.clamp(event_diffused, min=0.0, max=1.0)

        event_outs = self.forward(backbone_feats, t, event_diffused, event_gt, boundary_gt)

        decoder_ce_loss_total = 0
        decoder_mse_loss_total = 0
        decoder_boundary_loss_total = 0
        for e in event_outs:
            decoder_boundary = 1 - torch.einsum('bicl,bcjl->bijl',
                                                F.softmax(e[:, None, :, 1:], 2),
                                                F.softmax(e[:, :, None, :-1].detach(), 1)
                                                ).squeeze(1)

            if soft_label is None:  # To improve efficiency
                decoder_ce_loss = decoder_ce_criterion(
                    e.transpose(2, 1).contiguous().view(-1, self.num_classes),
                    torch.argmax(event_gt, dim=1).view(-1).long()  # batch_size must = 1
                )
            else:
                soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
                for i in range(soft_event_gt.shape[1]):
                    soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i],
                                                            soft_label)  # the soft label is not normalized
                soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)

                decoder_ce_loss = - soft_event_gt * F.log_softmax(e, 1)
                decoder_ce_loss = decoder_ce_loss.sum(0).sum(0)

            decoder_mse_loss = torch.clamp(decoder_mse_criterion(
                F.log_softmax(e[:, :, 1:], dim=1),
                F.log_softmax(e.detach()[:, :, :-1], dim=1)),
                min=0, max=16)

            decoder_boundary_loss = decoder_boundary_criterion(decoder_boundary, boundary_gt[:, :, 1:])
            decoder_boundary_loss = decoder_boundary_loss.mean()

            decoder_ce_loss = decoder_ce_loss.mean()
            decoder_mse_loss = decoder_mse_loss.mean()

            # 多个解码器输出结果的损失函数加和
            decoder_ce_loss_total += decoder_ce_loss
            decoder_mse_loss_total += decoder_mse_loss
            decoder_boundary_loss_total += decoder_mse_loss

        loss_dict = {
            'encoder_ce_loss_rgb': encoder_ce_loss_rgb,
            'encoder_mse_loss_rgb': encoder_mse_loss_rgb,
            'encoder_boundary_loss': encoder_boundary_loss,

            'encoder_ce_loss_flow': encoder_ce_loss_flow,
            'encoder_mse_loss_flow': encoder_mse_loss_flow,
            'encoder_boundary_loss': encoder_boundary_loss,

            'decoder_ce_loss': decoder_ce_loss_total,
            'decoder_mse_loss': decoder_mse_loss_total,
            'decoder_boundary_loss': decoder_boundary_loss_total,
        }

        return loss_dict

    @torch.no_grad()
    def ddim_sample(self, video_feats_rgb, video_feats_flow, proposal_prob=None, seed=None):

        if self.use_instance_norm:
            video_feats_rgb = self.ins_norm(video_feats_rgb)
            video_feats_flow = self.ins_norm(video_feats_flow)

        #         flops_encoder, _ = profile(self.encoder, inputs=(video_feats, ), verbose=False) #计算flops
        encoder_out_rgb, backbone_feats_rgb = self.encoder_rgb(video_feats_rgb, get_features=True)
        encoder_out_flow, backbone_feats_flow = self.encoder_flow(video_feats_flow, get_features=True)

        backbone_feats = self.cross_attn(backbone_feats_rgb, backbone_feats_flow)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # torch.Size([1, 19, 4847])
        shape = (video_feats_rgb.shape[0], self.num_classes, video_feats_rgb.shape[2])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))
        # [999, 749, 499, 249, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # [(999, 749), (749, 499), (499, 249), (249, -1)]

        if proposal_prob is not None:
            proposal_prob = proposal_prob.to(self.device).float()

            if proposal_prob.shape[-1] != shape[-1]:
                proposal_prob = F.interpolate(proposal_prob, size=shape[-1], mode='nearest')

            proposal_prob = torch.clamp(proposal_prob, min=1e-8)
            proposal_prob = proposal_prob / proposal_prob.sum(dim=1, keepdim=True)

            # proposal_prob: [0, 1] -> [-scale, scale]
            x_time = normalize(proposal_prob, self.scale)

            # 加一点噪声，让 diffusion 还有 refinement 空间
            x_time = x_time + 0.1 * torch.randn_like(x_time)
            x_time = torch.clamp(x_time, min=-self.scale, max=self.scale)
        else:
            x_time = torch.randn(shape, device=self.device)

        x_start = None
        #         flops = flops_encoder #计算flops
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)
            #             flops += flops_decoder    #计算flops
            x_return = torch.clone(x_start)

            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        x_return = denormalize(x_return, self.scale)
        #         print(flops/1e9)    #计算flops
        if seed is not None:
            t = 1000 * Time.time()  # current time in milliseconds
            t = int(t) % 2 ** 16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return


########################################################################################
# Encoder and Decoder are adapted from ASFormer.
# Compared to ASFormer, the main difference is that this version applies attention in a similar manner as dilated temporal convolutions.
# This difference does not change performance evidently in preliminary experiments.


class MultiModalCrossAttention(nn.Module):
    """单头交叉注意力核心模块（支持单向/双向）"""
    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.5, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.head_dim = head_dim

        # 模态A到模态B的投影层
        self.A_to_B_Q = nn.Conv1d(d_model, head_dim, 1)
        self.A_to_B_K = nn.Conv1d(d_model, head_dim, 1)
        self.A_to_B_V = nn.Conv1d(d_model, head_dim, 1)

        # 双向模式增加反向投影层
        if bidirectional:
            self.B_to_A_Q = nn.Conv1d(d_model, head_dim, 1)
            self.B_to_A_K = nn.Conv1d(d_model, head_dim, 1)
            self.B_to_A_V = nn.Conv1d(d_model, head_dim, 1)

        # 特征融合层
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model + head_dim, d_model, 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if bidirectional:
            self.fuse_b = nn.Sequential(
                nn.Conv1d(d_model + head_dim, d_model, 1),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor, 
                mask_a: torch.Tensor = None, mask_b: torch.Tensor = None):
        # 模态A→模态B注意力
        q_a = self.A_to_B_Q(feat_a)
        k_b = self.A_to_B_K(feat_b)
        v_b = self.A_to_B_V(feat_b)
        
        # 注意力计算
        attn_scores = torch.einsum('bct,bcs->bts', q_a, k_b) / (self.head_dim ** 0.5)
        if mask_b is not None:
            attn_scores = attn_scores.masked_fill(mask_b.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_b = torch.einsum('bts,bcs->bct', attn_weights, v_b)
        
        # 融合特征
        fused_a = self.fuse(torch.cat([feat_a, attended_b], dim=1))
        
        if not self.bidirectional:
            return fused_a
        
        # 模态B→模态A注意力（双向模式）
        q_b = self.B_to_A_Q(feat_b)
        k_a = self.B_to_A_K(feat_a)
        v_a = self.B_to_A_V(feat_a)
        
        attn_scores_ba = torch.einsum('bct,bcs->bts', q_b, k_a) / (self.head_dim ** 0.5)
        if mask_a is not None:
            attn_scores_ba = attn_scores_ba.masked_fill(mask_a.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights_ba = F.softmax(attn_scores_ba, dim=-1)
        attended_a = torch.einsum('bts,bcs->bct', attn_weights_ba, v_a)
        
        # 融合特征
        fused_b = self.fuse_b(torch.cat([feat_b, attended_a], dim=1))
        
        return fused_a, fused_b  # 保持返回两个独立特征

class AttentionLayer(nn.Module):
    """带残差连接和层归一化的单层注意力"""
    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.attn = MultiModalCrossAttention(d_model, head_dim, dropout, bidirectional)
        self.norm_a = nn.LayerNorm(d_model)  # 独立归一化层
        self.norm_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor, 
                mask_a: torch.Tensor = None, mask_b: torch.Tensor = None):
        # 保存原始特征用于残差连接
        ori_a, ori_b = feat_a, feat_b
        
        # 自动处理输入形状 [B, T, D] -> [B, D, T]
        if feat_a.size(-1) == self.attn.d_model:
            feat_a = feat_a.transpose(1, 2).contiguous()
            feat_b = feat_b.transpose(1, 2).contiguous()
        
        # 应用注意力
        attn_out = self.attn(feat_a, feat_b, mask_a, mask_b)
        
        # 残差连接与归一化
        if self.bidirectional:
            fused_a, fused_b = attn_out
            # 模态A残差连接
            fused_a = ori_a + self.dropout(fused_a)
            # 模态B残差连接
            fused_b = ori_b + self.dropout(fused_b)
            # 独立归一化
            return self.norm_a(fused_a.transpose(1, 2)).transpose(1, 2), self.norm_b(fused_b.transpose(1, 2)).transpose(1, 2)
        else:
            fused_a = ori_a + self.dropout(attn_out.transpose(1, 2))
            return self.norm_a(fused_a)

class MultiLayerAttention(nn.Module):
    """多层堆叠的交叉注意力网络（最终输出拼接）"""
    def __init__(self, num_layers: 2, d_model: int, head_dim: int, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionLayer(d_model, head_dim, dropout, bidirectional)
            for _ in range(num_layers)
        ])
        self.bidirectional = bidirectional
        self.final_concat = bidirectional  # 仅双向模式需要拼接
        self.downsample = nn.Conv1d(
                in_channels=2 * d_model,
                out_channels=d_model,
                kernel_size=1
            )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor, 
                mask_a: torch.Tensor = None, mask_b: torch.Tensor = None):
        # 逐层处理特征
        for layer in self.layers:
            if self.bidirectional:
                feat_a, feat_b = layer(feat_a, feat_b, mask_a, mask_b)
            else:
                feat_a = layer(feat_a, feat_b, mask_a, mask_b)
        
        # 核心修改：在最后一层后拼接双向输出
        if self.bidirectional:
            # 沿特征维度拼接 [B, T, 2*D]
            concat_output = torch.cat([feat_a, feat_b], dim=1)
            output = self.downsample(concat_output)
            return output
        return feat_a

class EncoderModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, kernel_size,
                 normal_dropout_rate, channel_dropout_rate, temporal_dropout_rate,
                 feature_layer_indices=None):
        super(EncoderModel, self).__init__()

        self.num_classes = num_classes
        self.feature_layer_indices = feature_layer_indices

        self.dropout_channel = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)

        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.encoder = MixedConvAttModule(num_layers, num_f_maps, kernel_size, normal_dropout_rate)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, get_features=False):
        if get_features:
            assert (self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0)
            features = []
            if -1 in self.feature_layer_indices:
                features.append(x)
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x, feature = self.encoder(self.conv_in(x), feature_layer_indices=self.feature_layer_indices)
            if feature is not None:
                features.append(feature)
            out = self.conv_out(x)
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))
            return out, torch.cat(features, 1)
        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            out = self.conv_out(self.encoder(self.conv_in(x), feature_layer_indices=None))
            return out


class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_classes,
                 num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate):
        super(DecoderModel, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, event):
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)

        fra = self.conv_in(event)
        fra = self.module(fra, x, time_emb)
        #         print(fra.shape)
        #         print(event.shape)
        event_out = self.conv_out(fra)

        return event_out


class MixedConvAttModuleV2(nn.Module):  # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  # 2 ** i

    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedConvAttentionLayerV2(nn.Module):

    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()

        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation

        assert (self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None

    def get_attn_indices(self, l, device):

        attn_indices = []

        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)
            # 3  3  3 ...                             (k=3, //1)
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)

            indices = [i + self.padding for i in range(s, e, step)]

            attn_indices.append(indices)

        attn_indices = np.array(attn_indices)

        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)

    def attention(self, x, x_cross):

        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)

        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2], :], (-1,))

        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)

        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1]))

        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)

        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2], :] >= self.padding,
            self.attn_indices[:x.shape[2], :] < att.shape[1] + self.padding
        )  # 1 keep, 0 mask

        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)

        return r

    def forward(self, x, x_cross):

        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)

        out = self.ffn_block(self.norm(out1 + out2))

        return x + out


class MixedConvAttModule(nn.Module):  # for encoder
    def __init__(self, num_layers, num_f_maps, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModule, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayer(num_f_maps, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  # 2 ** i

    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)

            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out


class MixedConvAttentionLayer(nn.Module):

    def __init__(self, d_model, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayer, self).__init__()

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation

        assert (self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None

    def get_attn_indices(self, l, device):

        attn_indices = []

        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)
            # 3  3  3 ...                             (k=3, //1)
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)

            indices = [i + self.padding for i in range(s, e, step)]

            attn_indices.append(indices)

        attn_indices = np.array(attn_indices)

        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)

    def attention(self, x):

        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)

        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2], :], (-1,))

        x_q = self.att_linear_q(x)
        x_k = self.att_linear_k(x)
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)

        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1]))

        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)

        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2], :] >= self.padding,
            self.attn_indices[:x.shape[2], :] < att.shape[1] + self.padding
        )  # 1 keep, 0 mask

        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)

        return r

    def forward(self, x):

        x_drop = self.dropout(x)
        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop)
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out