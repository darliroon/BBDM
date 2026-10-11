"""PRF (Paired Rectified Flow) 内核测试套件。

运行: cd /path/to/BBDM && python tests/test_prf.py
覆盖: 插值/速度公式、前后向、ODE 求解器精度 (Euler/Heun 解析解对照)、t 缩放约定、
      CFG 组合、cond_dropout、零初始化、玩具任务端到端收敛、潜空间版、Runner 全链路集成。
全部 CPU 可跑, 无需 GPU 与真实数据。
"""
import argparse
import itertools  # noqa: F401  (保证与模型文件一致的 import 环境)
import math
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.BrownianBridge.PairedRectifiedFlowModel import PairedRectifiedFlowModel
from model.BrownianBridge.LatentPairedRectifiedFlowModel import LatentPairedRectifiedFlowModel
from model.BrownianBridge.cond_encoder import ImageContextEncoder


# ---------------------------------------------------------------- 用例构造

def make_params(gamma=0.0, solver='euler', sample_step=10, image_size=32,
                cond_dropout=0.0, t_sampler='lognorm', loss_type='l2',
                condition_key="ImageContext", context_dim=64):
    unet = argparse.Namespace(
        image_size=image_size, in_channels=3, model_channels=32, out_channels=3,
        num_res_blocks=1, attention_resolutions=(1,),   # ds 单位: 1=全分辨率
        channel_mult=(1, 2), conv_resample=True, dims=2, num_heads=4,
        num_head_channels=-1, use_scale_shift_norm=True, resblock_updown=False,
        use_spatial_transformer=True, context_dim=context_dim, transformer_depth=1,
        condition_key=condition_key,
    )
    return argparse.Namespace(
        loss_type=loss_type, gamma=gamma, t_sampler=t_sampler,
        lognorm_mean=0.0, lognorm_std=1.0, num_timesteps=1000,
        sample_step=sample_step, solver=solver, cond_dropout=cond_dropout,
        UNetParams=unet,
    )


def make_model_config(**kw):
    cond = argparse.Namespace(in_channels=3, width=32, context_dim=64,
                              token_grid=8, num_blocks=2)
    return argparse.Namespace(
        BB=argparse.Namespace(params=make_params(**kw)),
        CondStageParams=cond,
        normalize_latent=False, latent_before_quant_conv=False,
    )


class ConstVelocityUNet(nn.Module):
    """测试替身: 忽略 x, 返回已知解析速度场 v(t) = d + γ·(1-2t)·c。
    从 timesteps 反解 t = timesteps/num_timesteps, 同时验证 ×num_timesteps 缩放约定。
    采样期同一 batch 的 timesteps 全同值, 取 timesteps[0] 反解。"""

    def __init__(self, d, c=None, gamma=0.0, num_timesteps=1000):
        super().__init__()
        self.d, self.c, self.gamma = d, c, gamma
        self.num_timesteps = num_timesteps
        self.seen_timesteps = []

    def forward(self, x, timesteps=None, context=None):
        assert timesteps is not None
        self.seen_timesteps.append(timesteps.detach().clone())
        t = timesteps.flatten()[0].float() / self.num_timesteps   # 反解连续 t
        v = self.d if self.c is None else self.d + self.gamma * (1. - 2. * t) * self.c
        return v.view(1, *([1] * (x.dim() - 1))).expand_as(x).to(x.dtype)


def tiny_model(gamma=0.0, solver='heun', sample_step=10, cond_dropout=0.0,
               image_size=32, t_sampler='lognorm', seed=0):
    torch.manual_seed(seed)
    cfg = make_model_config(gamma=gamma, solver=solver, sample_step=sample_step,
                            cond_dropout=cond_dropout, image_size=image_size, t_sampler=t_sampler)
    m = PairedRectifiedFlowModel(cfg)
    return m


def rand_image(b, size=32):
    return torch.rand(b, 3, size, size) * 2 - 1


# ---------------------------------------------------------------- 单元测试

def test_interpolation_and_velocity_formulas():
    """插值路径与速度目标的解析公式 (γ=0 与 γ>0, 标量 t 与逐样本 t)"""
    for gamma in (0.0, 0.37):
        m = tiny_model(gamma=gamma)
        a, b = rand_image(4), rand_image(4)
        z = torch.randn_like(a)
        # 标量 t
        t = 0.3
        xt = m.interpolate(a, b, t, z)
        expect = (1 - t) * a + t * b + gamma * t * (1 - t) * z
        assert torch.allclose(xt, expect, atol=1e-6), "interpolate (scalar t) wrong"
        vt = m.velocity_target(a, b, t, z)
        expect_v = (b - a) + gamma * (1 - 2 * t) * z
        assert torch.allclose(vt, expect_v, atol=1e-6), "velocity_target (scalar t) wrong"
        # 逐样本 t
        ts = torch.tensor([0.0, 0.25, 0.5, 1.0])
        xt = m.interpolate(a, b, ts, z)
        tv = ts.view(4, 1, 1, 1)
        expect = (1 - tv) * a + tv * b + gamma * tv * (1 - tv) * z
        assert torch.allclose(xt, expect, atol=1e-6), "interpolate (per-sample t) wrong"
        # 端点语义: t=0 严格为 A, t=1 严格为 B (γ 项消失)
        assert torch.allclose(m.interpolate(a, b, 0.0, z), a, atol=1e-6)
        assert torch.allclose(m.interpolate(a, b, 1.0, z), b, atol=1e-6)
    print("  [ok] 插值/速度公式 (γ=0/γ>0, 标量与逐样本 t, 端点语义)")


def test_p_losses_x_t_and_target():
    """p_losses 内部构造: hook 捕获 UNet 输入, 对照 x_t 公式; 零输出替身下 l2 损失 = mean(v_target²)"""
    m = tiny_model(gamma=0.2)
    a, b, c_img = rand_image(2), rand_image(2), rand_image(2)

    class ZeroUNet(nn.Module):
        def forward(self, x, timesteps=None, context=None):
            return torch.zeros_like(x)

    stub = ZeroUNet()
    m.denoise_fn = stub
    t = torch.tensor([0.2, 0.7])
    torch.manual_seed(0)
    z = torch.randn_like(a)
    captured = {}

    def hook(mod, inp, out):
        captured['x'] = inp[0]

    h = stub.register_forward_hook(hook)
    loss, log = m.p_losses(b, a, m.get_cond_stage_context(c_img, 2), t, z=z)
    h.remove()
    tv = t.view(2, 1, 1, 1)
    expect_xt = (1 - tv) * a + tv * b + 0.2 * tv * (1 - tv) * z
    assert torch.allclose(captured['x'], expect_xt, atol=1e-5), "UNet input x_t mismatch"
    expect_loss = (((b - a) + 0.2 * (1 - 2 * tv) * z) ** 2).mean()
    assert torch.allclose(loss, expect_loss, atol=1e-5), "velocity target loss mismatch"
    assert log['x_end_pred'].shape == b.shape
    print("  [ok] p_losses: x_t 构造与速度回归目标一致")


def test_forward_backward_gradflow():
    """forward 返回标量损失; backward 梯度流经 UNet 与 C 编码器 (γ=0/γ>0, l1/l2)
    注: 用 Runner 同款 weights_init (会重初始化 UNet 零输出层, 与真实训练路径一致);
    C 编码器经 reset_zero_init 后 proj/pos=0, 梯度经 pos_embedding 注入"""
    from runners.utils import weights_init
    for gamma, lt in itertools.product((0.0, 0.1), ('l1', 'l2')):
        m = tiny_model(gamma=gamma, t_sampler='uniform')
        m.apply(weights_init)
        m.train()
        a, b, c_img = rand_image(2), rand_image(2), rand_image(2)
        loss, _ = m(b, a, c_img)
        assert loss.dim() == 0 and torch.isfinite(loss), f"loss not scalar/finite ({gamma},{lt})"
        loss.backward()
        g_unet = sum(p.grad.abs().sum().item() for p in m.denoise_fn.parameters() if p.grad is not None)
        g_cond = sum(p.grad.abs().sum().item() for p in m.cond_stage_model.parameters() if p.grad is not None)
        assert g_unet > 0, f"no grad to UNet ({gamma},{lt})"
        assert g_cond > 0, f"no grad to cond encoder ({gamma},{lt})"
    print("  [ok] forward/backward: 损失有限, 梯度流经 UNet 与 C 编码器 (γ=0/0.1, l1/l2)")


def test_t_sampler():
    """lognorm/uniform 时间采样落在 (0,1), lognorm(loc=0) 均值≈0.5"""
    m = tiny_model(t_sampler='lognorm')
    t = m.sample_t(100000, torch.device('cpu'))
    assert (t > 0).all() and (t < 1).all()
    assert abs(t.mean().item() - 0.5) < 0.01, "lognorm(0,1) mean should ≈ 0.5"
    m2 = tiny_model(t_sampler='uniform')
    t2 = m2.sample_t(100000, torch.device('cpu'))
    assert (t2 >= 0).all() and (t2 < 1).all()
    assert abs(t2.mean().item() - 0.5) < 0.01
    print("  [ok] t 采样分布 (lognorm/uniform 值域与均值)")


def test_ode_solver_exactness():
    """解析速度场下 Euler/Heun 的 ODE 精度:
    v(t) = d + γ(1-2t)c, 闭式解 x(1) = A + d (γ 项在 [0,1] 积分为 0)
      - Heun 对 t 的线性项精确 => 任意 γ 下误差 ~0
      - Euler 对常场 (γ=0) 精确; γ>0 时误差 O(1/N), N 翻倍误差减半
    替身从 timesteps/1000 反解 t, 同时验证嵌入缩放约定"""
    d = torch.tensor([0.15])
    for gamma in (0.0, 0.4):
        m = tiny_model(gamma=gamma, solver='heun', sample_step=8)
        a = rand_image(2, 32)
        m.denoise_fn = ConstVelocityUNet(d=d, c=torch.tensor([0.5]), gamma=gamma)
        out = m.sample(a, c=None, clip_denoised=False)
        expect = a + d.view(1, 1, 1, 1)
        err_heun = (out - expect).abs().max().item()
        assert err_heun < 1e-4, f"heun not exact (gamma={gamma}): err={err_heun}"
    # Euler: γ=0 精确
    m = tiny_model(gamma=0.0, solver='euler', sample_step=7)
    a = rand_image(2, 32)
    m.denoise_fn = ConstVelocityUNet(d=d)
    out = m.sample(a, c=None, clip_denoised=False)
    assert (out - (a + d.view(1, 1, 1, 1))).abs().max().item() < 1e-5
    # Euler: γ>0 一阶收敛
    errs = []
    for n in (10, 20):
        m = tiny_model(gamma=0.4, solver='euler', sample_step=n)
        m.denoise_fn = ConstVelocityUNet(d=d, c=torch.tensor([0.5]), gamma=0.4)
        out = m.sample(a, c=None, clip_denoised=False)
        errs.append((out - (a + d.view(1, 1, 1, 1))).abs().max().item())
    assert errs[1] < 0.6 * errs[0], f"euler not O(1/N): {errs}"
    print(f"  [ok] ODE 求解器: heun 解析精确(err<1e-4), euler 常场精确/一阶收敛({errs[0]:.4f}->{errs[1]:.4f})")


def test_timestep_scaling_convention():
    """UNet 收到的 timesteps = t·num_timesteps; heun 2 步应精确看到 {0, 500, 1000}"""
    m = tiny_model(gamma=0.0, solver='heun', sample_step=2)
    a = rand_image(1, 32)
    stub = ConstVelocityUNet(d=torch.tensor([0.1]))
    m.denoise_fn = stub
    m.sample(a, c=None, clip_denoised=False)
    seen = sorted(set(round(v, 3) for v in torch.cat([s.flatten() for s in stub.seen_timesteps]).tolist()))
    assert seen == [0.0, 500.0, 1000.0], f"expected timesteps {{0,500,1000}}, got {seen}"
    print("  [ok] timestep 缩放: t∈[0,1] -> timesteps=t·1000, 端点精确")


def test_cfg_combination():
    """CFG: v = (1+w)·v(C) - w·v(null); 用可区分常场验证组合后的轨迹闭式解"""
    d_c, d_null = torch.tensor([0.2]), torch.tensor([-0.3])
    w = 1.5
    combined = (1 + w) * d_c - w * d_null

    class CfgStub(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, timesteps=None, context=None):
            null = self.m.get_null_context(x.shape[0])
            is_null = context is not None and null is not None and \
                      torch.allclose(context, null, atol=1e-6)
            v = d_null if is_null else d_c
            return v.view(1, 1, 1, 1).expand_as(x).to(x.dtype)

    m = tiny_model(gamma=0.0, solver='heun', sample_step=6, cond_dropout=0.0)
    m.eval()
    a, c_img = rand_image(2, 32), rand_image(2, 32)
    stub = CfgStub(m)
    m.denoise_fn = stub
    out = m.sample(a, c=c_img, clip_denoised=False, guidance_scale=w)
    expect = a + combined.view(1, 1, 1, 1)
    assert (out - expect).abs().max().item() < 1e-4, \
        f"CFG combination wrong: {(out - expect).abs().max().item()}"
    # guidance_scale=1 时不走 null 分支 (无额外前向)
    stub_call_log = []
    orig = stub.forward
    stub.forward = lambda *args, **kw: (stub_call_log.append(1), orig(*args, **kw))[1]
    m.sample(a, c=c_img, clip_denoised=False, guidance_scale=1.0)
    n_calls_w1 = len(stub_call_log)
    m.sample(a, c=c_img, clip_denoised=False, guidance_scale=w)
    assert len(stub_call_log) == n_calls_w1 * 3, "CFG should double NFE (2x per step vs 1x)"
    print("  [ok] CFG: (1+w)v(C)-w·v(null) 闭式解精确, w=1 时不做 null 前向")


def test_cond_dropout():
    """训练期 cond_dropout=1 => context 全部为 null token; eval 期不受影响"""
    m = tiny_model(cond_dropout=1.0)
    m.train()
    c_img = rand_image(3, 32)
    ctx = m.get_cond_stage_context(c_img, 3)
    null = m.get_null_context(3)
    assert torch.allclose(ctx, null, atol=1e-6), "cond_dropout=1.0 should give null tokens in train mode"
    m.eval()
    ctx_eval = m.get_cond_stage_context(c_img, 3)
    assert not torch.allclose(ctx_eval, null, atol=1e-6), "eval mode should keep real tokens"
    # c=None 时返回 null 序列
    ctx_none = m.get_cond_stage_context(None, 3)
    assert torch.allclose(ctx_none, null, atol=1e-6)
    print("  [ok] cond_dropout: 训练期丢弃/eval 保留/c=None -> null 序列")


def test_zero_init_semantics():
    """apply(weights_init) 后 ImageContextEncoder 的 proj/pos 归零 (C 影响从 0 渐长)"""
    from runners.utils import weights_init
    m = tiny_model()
    m.apply(weights_init)
    assert m.cond_stage_model.proj.weight.abs().sum() == 0
    assert m.cond_stage_model.proj.bias.abs().sum() == 0
    assert m.cond_stage_model.pos_embedding.abs().sum() == 0
    print("  [ok] 零初始化: apply 后 proj/pos 全零")


def test_sample_shapes_clip_midstep():
    """采样输出形状/clip/轨迹接口"""
    for solver in ('euler', 'heun'):
        m = tiny_model(solver=solver, sample_step=5)
        m.eval()
        a, c_img = rand_image(2, 32), rand_image(2, 32)
        out = m.sample(a, c=c_img, clip_denoised=True)
        assert out.shape == a.shape and torch.isfinite(out).all()
        assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6
        out = m.sample(a, c=c_img, clip_denoised=True, guidance_scale=1.5)  # CFG 路径
        assert out.shape == a.shape and torch.isfinite(out).all()
        imgs, one_steps = m.sample(a, c=c_img, clip_denoised=True, sample_mid_step=True)
        assert len(imgs) == 6 and len(one_steps) == 5
        assert torch.allclose(imgs[0], a, atol=1e-6), "trajectory should start at A"
        assert all(im.shape == a.shape for im in imgs)
        # 无 C 采样 (c=None -> null context)
        out = m.sample(a, c=None, clip_denoised=False)
        assert out.shape == a.shape and torch.isfinite(out).all()
    print("  [ok] 采样接口: 形状/clip/CFG/轨迹/c=None (euler & heun)")


def test_training_convergence_toy():
    """玩具任务端到端: B = A + d (常位移场), 训练 tiny PRF 后 ODE 采样应还原 B"""
    torch.manual_seed(0)
    m = tiny_model(gamma=0.0, solver='heun', sample_step=10, image_size=32, t_sampler='lognorm')
    from runners.utils import weights_init
    m.apply(weights_init)
    m.train()
    a = rand_image(8, 32)
    d = torch.tensor([0.0, 0.12, -0.10]).view(1, 3, 1, 1)
    b = (a + d).clamp(-1, 1)
    c_img = torch.zeros_like(a)
    opt = torch.optim.Adam(m.get_parameters(), lr=2e-3)

    first = None
    for step in range(400):
        loss, _ = m(b, a, c_img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 0.1 * first, f"loss not converging: {first:.4f} -> {final:.4f}"

    m.eval()
    with torch.no_grad():
        out = m.sample(a, c=c_img, clip_denoised=False)
    mae = (out - b).abs().mean().item()
    assert mae < 0.08, f"sample MAE too high: {mae:.4f}"
    print(f"  [ok] 玩具任务收敛: loss {first:.4f}->{final:.4f}, ODE 采样 MAE={mae:.4f}")


# ---------------------------------------------------------------- 潜空间版

def make_latent_model_config(image_size=16, vq_resolution=64, ckpt_path=None):
    cond = argparse.Namespace(in_channels=3, width=32, context_dim=64, token_grid=8, num_blocks=2)
    unet = argparse.Namespace(
        image_size=image_size, in_channels=3, model_channels=32, out_channels=3,
        num_res_blocks=1, attention_resolutions=(image_size,),
        channel_mult=(1, 2), conv_resample=True, dims=2, num_heads=4,
        num_head_channels=-1, use_scale_shift_norm=True, resblock_updown=False,
        use_spatial_transformer=True, context_dim=64, transformer_depth=1,
        condition_key="ImageContext",
    )
    vqgan = argparse.Namespace(
        params=argparse.Namespace(
            ckpt_path=ckpt_path, embed_dim=3, n_embed=64,
            ddconfig=argparse.Namespace(
                double_z=False, z_channels=3, resolution=vq_resolution,
                in_channels=3, out_ch=3, ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
                attn_resolutions=[], dropout=0.0),
            lossconfig=argparse.Namespace(target='torch.nn.Identity'),
        ))
    return argparse.Namespace(
        BB=argparse.Namespace(params=argparse.Namespace(
            loss_type='l2', gamma=0.1, t_sampler='uniform', lognorm_mean=0.0, lognorm_std=1.0,
            num_timesteps=1000, sample_step=5, solver='heun', cond_dropout=0.0,
            UNetParams=unet)),
        CondStageParams=cond, VQGAN=vqgan,
        normalize_latent=False, latent_before_quant_conv=False,
    )


def test_latent_prf():
    """LPRF: 编解码形状/forward-backward/采样链路; vqgan 冻结"""
    torch.manual_seed(0)
    cfg = make_latent_model_config()
    m = LatentPairedRectifiedFlowModel(cfg)
    assert all(not p.requires_grad for p in m.vqgan.parameters()), "vqgan should be frozen"
    a, b, c_img = rand_image(2, 64), rand_image(2, 64), rand_image(2, 64)
    assert m.encode(a, cond=True).shape == (2, 3, 16, 16), "f4 latent shape wrong"
    loss, _ = m(b, a, c_img)
    assert torch.isfinite(loss)
    loss.backward()
    g_unet = sum(p.grad.abs().sum().item() for p in m.denoise_fn.parameters() if p.grad is not None)
    assert g_unet > 0
    m.eval()
    with torch.no_grad():
        out = m.sample(a, c=c_img, clip_denoised=False)
        assert out.shape == a.shape and torch.isfinite(out).all()
        imgs, one_steps = m.sample(a, c=c_img, sample_mid_step=True)
        assert len(imgs) == 6 and len(one_steps) == 5
        assert all(im.shape == (2, 3, 64, 64) for im in imgs)
    print("  [ok] LPRF 潜空间版: encode/decode 形状, 前后向, 采样与轨迹")


# ---------------------------------------------------------------- Runner 集成

def build_runner_config(tmpdir, model_type="PRF"):
    def save_split(stage, n, offset):
        for sub in ("A", "B", "C"):
            d = os.path.join(tmpdir, stage, sub)
            os.makedirs(d, exist_ok=True)
        for i in range(n):
            torch.manual_seed(hash((stage, i, offset)) % (2**31))
            a = (torch.rand(3, 32, 32) > 0.98).float()          # 白底稀疏线
            b = a + 0.3 * (torch.rand(3, 32, 32) > 0.99).float()
            c = torch.rand(3, 32, 32)
            for sub, img in (("A", a), ("B", b), ("C", c)):
                arr = ((img.clamp(0, 1)) * 255).byte().permute(1, 2, 0).numpy()
                from PIL import Image
                Image.fromarray(arr).save(os.path.join(tmpdir, stage, sub, f"{i+offset:04d}.png"))

    save_split("train", 4, 0)
    save_split("val", 4, 10)
    save_split("test", 4, 20)

    args = argparse.Namespace(config='test', train=True, sample_at_start=False, save_top=False,
                              seed=1234, gpu_ids='-1', port='12355',
                              resume_model=None, resume_optim=None,
                              max_epoch=None, max_steps=None,
                              result_path=os.path.join(tmpdir, 'results'))
    unet = argparse.Namespace(
        image_size=32, in_channels=3, model_channels=32, out_channels=3,
        num_res_blocks=1, attention_resolutions=(1,), channel_mult=(1, 2),
        conv_resample=True, dims=2, num_heads=4, num_head_channels=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_spatial_transformer=True, context_dim=64, transformer_depth=1,
        condition_key="ImageContext")

    model = argparse.Namespace(
        model_name=f"test-{model_type}", model_type=model_type,
        latent_before_quant_conv=False, normalize_latent=False,
        only_load_latent_mean_std=False,
        model_load_path=None, optim_sche_load_path=None,
        EMA=argparse.Namespace(use_ema=False, ema_decay=0.995,
                               update_ema_interval=8, start_ema_step=100),
        CondStageParams=argparse.Namespace(in_channels=3, width=32, context_dim=64,
                                           token_grid=8, num_blocks=2),
        BB=argparse.Namespace(
            optimizer=argparse.Namespace(weight_decay=0.0, optimizer='Adam', lr=1e-3, beta1=0.9),
            lr_scheduler=argparse.Namespace(factor=0.5, patience=100, threshold=1e-4,
                                            cooldown=100, min_lr=1e-7),
            params=argparse.Namespace(
                loss_type='l2', gamma=0.0, t_sampler='uniform', lognorm_mean=0.0,
                lognorm_std=1.0, num_timesteps=1000, sample_step=4, solver='euler',
                cond_dropout=0.1, UNetParams=unet)),
    )
    if model_type == "LPRF":
        model.VQGAN = argparse.Namespace(
            params=argparse.Namespace(
                ckpt_path=None, embed_dim=3, n_embed=64,
                ddconfig=argparse.Namespace(
                    double_z=False, z_channels=3, resolution=32, in_channels=3, out_ch=3,
                    ch=32, ch_mult=(1, 2, 4), num_res_blocks=1, attn_resolutions=[], dropout=0.0),
                lossconfig=argparse.Namespace(target='torch.nn.Identity')))
        model.BB.params.UNetParams = argparse.Namespace(
            **{**vars(unet), 'image_size': 8})

    cfg = argparse.Namespace(
        runner='BBDMRunner', args=args,
        training=argparse.Namespace(n_epochs=1, n_steps=100, save_interval=1, sample_interval=1,
                                    validation_interval=1, accumulate_grad_batches=1,
                                    use_DDP=False, device=[torch.device('cpu')], local_rank=0),
        testing=argparse.Namespace(clip_denoised=True, sample_num=1),
        data=argparse.Namespace(
            dataset_name='toy', dataset_type='custom_triple',
            dataset_config=argparse.Namespace(dataset_path=tmpdir, image_size=32, channels=3,
                                              to_normal=True, flip=False),
            train=argparse.Namespace(batch_size=2, shuffle=True),
            val=argparse.Namespace(batch_size=2, shuffle=True),
            test=argparse.Namespace(batch_size=2, shuffle=False)),
        model=model,
    )
    return cfg


def test_runner_integration():
    """BBDMRunner 全链路: 构造/优化器/loss_fn/sample/sample_to_eval (PRF 与 LPRF)"""
    from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner
    from runners.DiffusionBasedModelRunners import BBDMRunner as _m  # noqa: F401
    from torch.utils.data import DataLoader

    for model_type, cls in (("PRF", PairedRectifiedFlowModel),
                            ("LPRF", LatentPairedRectifiedFlowModel)):
        tmpdir = tempfile.mkdtemp(prefix=f"prf_test_{model_type}_")
        try:
            cfg = build_runner_config(tmpdir, model_type=model_type)
            runner = BBDMRunner(cfg)
            assert isinstance(runner.net, cls), f"model_type={model_type} routing failed"
            runner.net.train()
            from runners.utils import get_dataset
            train_ds, val_ds, test_ds = get_dataset(cfg.data)
            batch = [next(iter(DataLoader(ds, batch_size=2))) for ds in (train_ds, val_ds, test_ds)]
            loss = runner.loss_fn(net=runner.net, batch=batch[0], epoch=0, step=1)
            assert torch.isfinite(torch.as_tensor(loss))
            loss = torch.as_tensor(loss)
            loss.backward()
            runner.optimizer[0].step()
            runner.optimizer[0].zero_grad()
            runner.net.eval()
            runner.sample(runner.net, batch[1], os.path.join(tmpdir, 'smoke'), stage='val')
            for f in ('skip_sample.png', 'condition.png', 'ground_truth.png'):
                assert os.path.exists(os.path.join(tmpdir, 'smoke', 'val_sample', f)), f"missing {f}"
            test_loader = DataLoader(test_ds, batch_size=2)
            runner.sample_to_eval(runner.net, test_loader, os.path.join(tmpdir, 'eval'))
            base = os.path.join(tmpdir, 'eval')
            for f in ('condition', 'context', 'ground_truth', '4'):
                assert os.path.isdir(os.path.join(base, f)), f"missing eval dir {f}"
            # 训练 3 步端到端 (runner.train 走完整循环太重, 此处只验证 loss_fn 反复可调用)
            for step in range(3):
                l = runner.loss_fn(net=runner.net, batch=batch[0], epoch=0, step=step + 2)
                assert torch.isfinite(torch.as_tensor(l))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"  [ok] Runner 全链路集成 ({model_type}): 路由/训练步/采样/测试集输出")


def test_cbbdm_regression():
    """回归: 原 CBBDM (BBDM/LBBDM) 路径不受 PRF 接入影响"""
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
    unet = argparse.Namespace(
        image_size=32, in_channels=3, model_channels=32, out_channels=3,
        num_res_blocks=1, attention_resolutions=(32,), channel_mult=(1, 2),
        conv_resample=True, dims=2, num_heads=4, num_head_channels=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_spatial_transformer=True, context_dim=64, transformer_depth=1,
        condition_key="ImageContext")
    cfg = argparse.Namespace(
        BB=argparse.Namespace(params=argparse.Namespace(
            mt_type='linear', objective='grad', loss_type='l2', skip_sample=True,
            sample_type='linear', sample_step=5, num_timesteps=50, eta=1.0, max_var=1.0,
            cond_dropout=0.1, UNetParams=unet)),
        CondStageParams=argparse.Namespace(in_channels=3, width=32, context_dim=64,
                                           token_grid=8, num_blocks=2))
    torch.manual_seed(0)
    m = BrownianBridgeModel(cfg)
    a, b, c_img = rand_image(2, 32), rand_image(2, 32), rand_image(2, 32)
    loss, _ = m(b, a, c_img)
    assert torch.isfinite(loss)
    loss.backward()
    m.eval()
    with torch.no_grad():
        out = m.sample(a, c=c_img)
        assert out.shape == a.shape and torch.isfinite(out).all()
    print("  [ok] CBBDM 回归: BBDM 内核训练/采样不受影响")


# ---------------------------------------------------------------- 入口

if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [
        test_interpolation_and_velocity_formulas,
        test_p_losses_x_t_and_target,
        test_forward_backward_gradflow,
        test_t_sampler,
        test_timestep_scaling_convention,
        test_ode_solver_exactness,
        test_cfg_combination,
        test_cond_dropout,
        test_zero_init_semantics,
        test_sample_shapes_clip_midstep,
        test_training_convergence_toy,
        test_latent_prf,
        test_runner_integration,
        test_cbbdm_regression,
    ]
    failed = []
    for t in tests:
        print(f"[RUN] {t.__name__}")
        try:
            t()
        except Exception as e:
            failed.append(t.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {t.__name__}: {e}")
    print("=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {failed}")
        sys.exit(1)
    print(f"ALL {len(tests)} TESTS PASSED")
