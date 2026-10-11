import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.cond_encoder import ImageContextEncoder


class PairedRectifiedFlowModel(nn.Module):
    """Paired Rectified Flow (PRF) 内核: A -> B 变换 + C 条件注入。

    与 BrownianBridgeModel (CBBDM) 同接口 (forward / sample / get_parameters / apply),
    可作为可替换内核接入 BBDMRunner, 数据管线与 CondStageParams 完全复用。

    数学 (stochastic interpolants, Albergo & Vanden-Eijnden 2023):
        插值路径   x_t = (1-t)·A + t·B + γ·t(1-t)·z,   z ~ N(0, I)  (整条路径共享同一 z)
        速度目标   v*(x_t, t) = E[ẋ_t | x_t]           (ODE 速度场, CFM 回归无偏)
        训练目标   u_t   = (B - A) + γ·(1-2t)·z        (ẋ_t 本身, 期望意义下等于 v*)
        采样       x(0) = A, dx/dt = v̂(x, t), Euler/Heun 积分到 t=1 得 B̂
    γ = 0 时退化为 Rectified Flow (Liu et al. 2023) / paired interpolation;
    γ > 0 时路径带噪声正则 (起点终点仍严格为 A / B, 因 γ·t(1-t) 在 t=0,1 处为 0),
    ODE 采样仍是确定性的 —— 训练路径的随机性只作为正则, 不改变采样过程。

    工程约定与 SD3 / Flux (diffusers FlowMatchEulerDiscreteScheduler) 一致:
        - 连续 t ∈ (0, 1), UNet timestep 嵌入输入 t·num_timesteps (支持小数)
        - 时间步采样: logit-normal (SD3 默认) 或 uniform
        - 求解器: Euler (1 阶) / Heun (2 阶)
        - CFG: v = (1+w)·v(C) - w·v(null), 与 CBBDM 相同

    配置读取 model.BB.params (与 CBBDM 共用 optimizer/调度器配置段):
        loss_type:     'l1' | 'l2'          速度回归损失
        gamma:         float, 默认 0.0      插值噪声强度 γ
        t_sampler:     'lognorm' | 'uniform' 时间步分布
        lognorm_mean/std: logit-normal 参数 (loc, scale), 默认 (0.0, 1.0)
        num_timesteps: int, 默认 1000       timestep 嵌入尺度 (t·num_timesteps), 与 CBBDM UNet 一致
        sample_step:   int, 默认 50         采样 ODE 步数
        solver:        'euler' | 'heun'
        cond_dropout:  float, 默认 0.0      C 条件丢弃概率 (CFG 训练)
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.BB.params

        # PRF 超参 (带默认值, CBBDM 模板参数可直接沿用)
        self.loss_type = model_params.loss_type if model_params.__contains__("loss_type") else 'l1'
        self.gamma = model_params.gamma if model_params.__contains__("gamma") else 0.0
        self.t_sampler = model_params.t_sampler if model_params.__contains__("t_sampler") else 'lognorm'
        self.lognorm_mean = model_params.lognorm_mean if model_params.__contains__("lognorm_mean") else 0.0
        self.lognorm_std = model_params.lognorm_std if model_params.__contains__("lognorm_std") else 1.0
        # timestep 嵌入尺度: 连续 t∈[0,1] 映射到 [0, num_timesteps], 与 CBBDM UNet 的
        # 正弦嵌入频率范围一致 (SD3/Flux 同款约定, timestep_embedding 支持小数)
        self.num_timesteps = model_params.num_timesteps if model_params.__contains__("num_timesteps") else 1000
        self.sample_step = model_params.sample_step if model_params.__contains__("sample_step") else 50
        self.solver = model_params.solver if model_params.__contains__("solver") else 'euler'
        assert self.solver in ('euler', 'heun'), f"unknown solver {self.solver}"
        assert self.sample_step >= 1, "sample_step must be >= 1"
        assert self.t_sampler in ('lognorm', 'uniform'), f"unknown t_sampler {self.t_sampler}"
        assert self.gamma >= 0.0, "gamma must be >= 0"

        # UNet / 条件编码器: 与 BrownianBridgeModel 完全一致
        self.image_size = model_params.UNetParams.image_size
        self.condition_key = model_params.UNetParams.condition_key
        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

        self.cond_dropout = getattr(model_params, "cond_dropout", 0.0)
        if self.condition_key == "ImageContext":
            cond_params = getattr(model_config, "CondStageParams", None)
            assert cond_params is not None, \
                "condition_key=ImageContext 需要在 model.CondStageParams 配置 ImageContextEncoder 参数"
            self.cond_stage_model = ImageContextEncoder(**vars(cond_params))
        else:
            self.cond_stage_model = None

    # ---------------- 权重初始化 / 参数 (与 CBBDM 对齐) ----------------

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        cond_stage = getattr(self, "cond_stage_model", None)
        if isinstance(cond_stage, ImageContextEncoder):
            cond_stage.apply(weight_init)
            cond_stage.reset_zero_init()
        return self

    def get_parameters(self):
        if getattr(self, "cond_stage_model", None) is not None:
            return itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        return self.denoise_fn.parameters()

    # ---------------- 条件注入 (与 CBBDM 完全一致) ----------------

    def get_cond_stage_context(self, c, batch_size=None):
        """图C -> cross-attention token 序列; 训练期按 cond_dropout 整样本丢弃 (CFG)"""
        if self.condition_key != "ImageContext" or self.cond_stage_model is None:
            return None
        if c is None:
            return self.cond_stage_model.null_tokens(batch_size if batch_size is not None else 1)
        drop_prob = self.cond_dropout if self.training else 0.0
        return self.cond_stage_model(c, drop_prob=drop_prob)

    def get_null_context(self, batch_size):
        if self.condition_key != "ImageContext" or self.cond_stage_model is None:
            return None
        return self.cond_stage_model.null_tokens(batch_size)

    # ---------------- 插值与速度目标 (纯函数, 便于测试) ----------------

    def interpolate(self, a, b, t, z):
        """x_t = (1-t)·A + t·B + γ·t(1-t)·z;  t: (B,) 或标量"""
        t_ = self._expand(t, a)
        return (1. - t_) * a + t_ * b + self.gamma * t_ * (1. - t_) * z

    def velocity_target(self, a, b, t, z):
        """u_t = ẋ_t = (B - A) + γ·(1-2t)·z;  t: (B,) 或标量"""
        t_ = self._expand(t, a)
        return (b - a) + self.gamma * (1. - 2. * t_) * z

    @staticmethod
    def _expand(t, x):
        """t: (B,) tensor -> 广播形状; 标量 -> 原样 (算术运算自动广播)"""
        if torch.is_tensor(t) and t.dim() > 0:
            b = t.shape[0]
            return t.view(b, *((1,) * (x.dim() - 1)))
        return t

    def sample_t(self, batch_size, device):
        """时间步采样: logit-normal (SD3 默认) 或 uniform"""
        if self.t_sampler == 'lognorm':
            u = self.lognorm_mean + self.lognorm_std * torch.randn(batch_size, device=device)
            return torch.sigmoid(u)
        return torch.rand(batch_size, device=device)

    # ---------------- 训练 ----------------

    def forward(self, x, y, c=None, context=None):
        """x: 目标图 B;  y: 源图 A (ODE 起点);  c: 条件图 C"""
        if self.condition_key == "nocond":
            context = None
        elif context is None:
            if self.condition_key == "ImageContext":
                context = self.get_cond_stage_context(c, batch_size=x.shape[0])
            else:
                context = y
        b, c_ch, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = self.sample_t(b, device)
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, z=None):
        """
        速度回归损失
        :param x0: 目标 B
        :param y: 源 A
        :param t: (B,) 时间步
        :param z: 共享高斯噪声 (可选, 测试用)
        """
        b, c, h, w = x0.shape
        z = z if z is not None else torch.randn_like(x0)

        x_t = self.interpolate(y, x0, t, z)
        v_target = self.velocity_target(y, x0, t, z)

        # 连续 t -> UNet timestep 嵌入尺度 (SD3/Flux 约定, 支持小数)
        t_emb = t.float() * self.num_timesteps
        v_recon = self.denoise_fn(x_t, timesteps=t_emb, context=context)

        if self.loss_type == 'l1':
            loss = (v_recon - v_target).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(v_recon, v_target)
        else:
            raise NotImplementedError()

        # 终点估计 (诊断/可视化用, 类比 CBBDM 的 x0_recon): 从 x_t 一步直达 t=1
        with torch.no_grad():
            t_ = t.view(b, *((1,) * (x0.dim() - 1)))
            x_end_pred = x_t + v_recon * (1. - t_)
        log_dict = {
            "loss": loss,
            "x_end_pred": x_end_pred
        }
        return loss, log_dict

    # ---------------- 采样 (ODE) ----------------

    def _velocity(self, x_t, t, context, context_null=None, guidance_scale=1.0):
        """v̂(x_t, t); t: 标量 float; 可选 CFG 组合"""
        b = x_t.shape[0]
        t_emb = torch.full((b,), float(t) * self.num_timesteps, device=x_t.device, dtype=torch.float32)
        v = self.denoise_fn(x_t, timesteps=t_emb, context=context)
        if context_null is not None:
            v_null = self.denoise_fn(x_t, timesteps=t_emb, context=context_null)
            v = (1.0 + guidance_scale) * v - guidance_scale * v_null
        return v

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False,
                      context_null=None, guidance_scale=1.0):
        """从 y=A 出发积分 ODE 到 t=1 (Euler / Heun)"""
        if self.condition_key == "nocond":
            context = None
        elif context is None and self.condition_key != "ImageContext":
            context = y

        n = self.sample_step
        ts = torch.linspace(0.0, 1.0, n + 1)
        x = y
        imgs, one_step_imgs = ([y], []) if sample_mid_step else (None, None)
        for i in tqdm(range(n), desc='ODE sampling loop', total=n, disable=not sample_mid_step):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            dt = t1 - t0
            if self.solver == 'euler':
                v = self._velocity(x, t0, context, context_null, guidance_scale)
                if sample_mid_step:
                    # 一步终点估计 (类比 CBBDM 每步的 x0_recon): 从当前点直达 t=1
                    one_step_imgs.append(x + v * (1. - t0))
                x = x + v * dt
            else:  # heun
                v1 = self._velocity(x, t0, context, context_null, guidance_scale)
                if sample_mid_step:
                    one_step_imgs.append(x + v1 * (1. - t0))
                v2 = self._velocity(x + v1 * dt, t1, context, context_null, guidance_scale)
                x = x + 0.5 * (v1 + v2) * dt
            if sample_mid_step:
                imgs.append(x)
        if sample_mid_step:
            if clip_denoised:
                imgs = [imgs[0]] + [im.clamp(-1., 1.) for im in imgs[1:]]
                one_step_imgs = [im.clamp(-1., 1.) for im in one_step_imgs]
            return imgs, one_step_imgs
        if clip_denoised:
            x = x.clamp(-1., 1.)
        return x

    @torch.no_grad()
    def sample(self, y, c=None, clip_denoised=True, sample_mid_step=False, guidance_scale=1.0):
        """y: 源图 A;  c: 条件图 C;  返回 B̂ (或 (轨迹, 终点估计) 列表)"""
        context = self.get_cond_stage_context(c, batch_size=y.shape[0])
        context_null = self.get_null_context(y.shape[0]) \
            if (guidance_scale != 1.0 and context is not None) else None
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step,
                                  context_null=context_null, guidance_scale=guidance_scale)
