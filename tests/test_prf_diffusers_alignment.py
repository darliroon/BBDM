"""PRF 内核与 diffusers FlowMatchEulerDiscreteScheduler 的数值对齐测试。

彻底闭环验证 PRF 的三处核心约定与 diffusers (SD3/Flux 生态, 0.41.0) 完全一致:
  1. 训练目标 (速度场公式, 含 γ 噪声项)   —— 有限差分独立验证
  2. timestep 嵌入缩放 (t×1000) 与 sigma 网格端点
  3. Euler 采样器 (更新公式/方向/符号)     —— 解析场与真实 UNet 双重对齐

两套约定的映射关系 (测试的理论基础):
  diffusers/SD3:  x_σ = (1-σ)·data + σ·noise,  v = dx/dσ = noise - data,
                  采样 σ: 1→0 (从 noise 走向 data), timestep = σ·1000
  PRF:            x_t = (1-t)·A + t·B + γ·t(1-t)·z,  u = (B-A) + γ(1-2t)·z,
                  采样 t: 0→1 (从 A 走向 B), timestep = t·1000
  映射 (data↔B, noise↔A):  σ = 1 - t,  v_diffusers(σ) = -u_PRF(1-σ)
  代入验证: x_{t=1-σ} = σ·A + (1-σ)·B + γσ(1-σ)·z
            dx/dσ = A - B + γ(1-2σ)·z = -[(B-A) + γ(1-2(1-σ))·z] = -u_PRF(1-σ)  ✓

运行: cd /path/to/BBDM && python tests/test_prf_diffusers_alignment.py
依赖: pip install diffusers --no-deps (缺失时本测试自动 SKIP, 不影响主套件)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

from test_prf import tiny_model, rand_image  # noqa: E402


def make_scheduler(n_steps, num_train_timesteps=1000):
    """构造与 PRF linspace(0,1,n+1) 网格完全对齐的调度器。

    PRF 网格: ts = [0, 1/n, ..., 1] (升序)
    diffusers sigma 网格 (降序, 末尾自动补 0): [1, 1-1/n, ..., 1/n] + [0]
    通过 set_timesteps 的 sigmas 参数注入; shift=1.0 时变换恒等。"""
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps, shift=1.0)
    sigmas_desc = [1.0 - i / n_steps for i in range(n_steps)]  # [1, 1-1/n, ..., 1/n]
    sch.set_timesteps(n_steps, sigmas=sigmas_desc)
    return sch


# ---------------------------------------------------------------- 测试

def test_fd_velocity_in_sigma_space():
    """有限差分独立验证: PRF 速度目标在 σ 空间恰为 diffusers 约定的 -(dx/dσ)。
    即 u_PRF(t) = -dx/dσ |_{σ=1-t}, 对 γ>0 成立 (含噪声项导数)。"""
    gamma = 0.4
    m = tiny_model(gamma=gamma)
    a, b = rand_image(2).double(), rand_image(2).double()
    z = torch.randn_like(a)
    h = 1e-4
    for sigma in (0.15, 0.5, 0.83):
        t = 1.0 - sigma
        # x(σ) = x_{t=1-σ}: 用 PRF 自己的 interpolate 构造
        x_hi = m.interpolate(a, b, t - h, z)   # σ+h 方向 = t-h
        x_lo = m.interpolate(a, b, t + h, z)   # σ-h 方向 = t+h
        dx_dsigma_fd = (x_hi - x_lo) / (2 * h)
        # PRF 速度目标取负应等于 dx/dσ
        u = m.velocity_target(a, b, t, z)
        err = (dx_dsigma_fd - (-u)).abs().max().item()
        assert err < 1e-5, f"FD mismatch at σ={sigma}: {err}"
    print("  [ok] FD 验证: dx/dσ = -u_PRF(1-σ), γ=0.4 含噪声项导数一致")


def test_sigma_grid_and_timestep_convention():
    """网格与 timestep 缩放约定: sigma 首尾严格 (1, 0) ↔ PRF t 首尾 (0, 1);
    timestep = σ·1000 与 PRF t_emb = t·1000 同一缩放。"""
    for n in (8, 13):
        sch = make_scheduler(n)
        sig = sch.sigmas
        assert len(sig) == n + 1, f"sigmas should have n+1 entries, got {len(sig)}"
        assert abs(float(sig[0]) - 1.0) < 1e-6 and abs(float(sig[-1])) < 1e-6, \
            f"sigma endpoints should be (1, 0), got ({float(sig[0])}, {float(sig[-1])})"
        # timestep = sigma * num_train_timesteps (diffusers 约定 = PRF t×1000 的镜像)
        ts_diffusers = sch.timesteps
        assert abs(float(ts_diffusers[0]) - 1000.0) < 1e-3, "first timestep should be 1000"
        # 中点对照: sigma[i] = 1-i/n ↔ PRF t = i/n ↔ t_emb = i/n*1000
        i = n // 2
        t_prf_emb = (i / n) * 1000
        t_diffusers = float(ts_diffusers[i])
        assert abs((1000.0 - t_diffusers) - t_prf_emb) < 1e-2, \
            f"timestep mirror mismatch: diffusers={t_diffusers}, prf={t_prf_emb}"
    print("  [ok] sigma 网格端点 (1,0) ↔ t 端点 (0,1); timestep=σ·1000 镜像一致")


class DiffusersWrapper(torch.nn.Module):
    """把 PRF 的 UNet 包装成 diffusers 调度器期望的模型:
    收到 τ=σ·1000 -> 换算 PRF 时间 t=1-τ/1000 -> 调 PRF UNet (t_emb=t·1000) -> 取负。"""

    def __init__(self, prf_model, context):
        super().__init__()
        self.m = prf_model
        self.ctx = context

    def forward(self, x, timesteps=None, context=None):
        t = 1.0 - float(timesteps.flatten()[0]) / self.m.num_timesteps
        t_emb = torch.full((x.shape[0],), t * self.m.num_timesteps,
                           device=x.device, dtype=torch.float32)
        v = self.m.denoise_fn(x, timesteps=t_emb, context=self.ctx)
        return -v


def run_scheduler(sch, model, x0):
    """驱动 diffusers 调度器循环, 返回终点样本 (对标 PRF 的 p_sample_loop)"""
    x = x0.clone()
    for i in range(len(sch.timesteps)):
        tau = sch.timesteps[i]
        out = model(x, timesteps=tau)
        x = sch.step(out, tau, x).prev_sample
    return x


def test_euler_analytic_alignment():
    """解析速度场下 Euler 逐步对齐 (γ=0 与 γ>0, N=8 与 N=13)。
    两边同网格同算法, 应逐步 allclose; γ=0 另对照闭式解 A+d。"""
    from test_prf import ConstVelocityUNet
    d = torch.tensor([0.15])
    c = torch.tensor([0.5])
    for gamma in (0.0, 0.4):
        for n in (8, 13):
            m = tiny_model(gamma=gamma, solver='euler', sample_step=n)
            m.eval()
            m.denoise_fn = ConstVelocityUNet(d=d, c=c if gamma > 0 else None, gamma=gamma)
            a = rand_image(2, 32)
            ours = m.sample(a, c=None, clip_denoised=False)
            sch = make_scheduler(n)
            wrapped = DiffusersWrapper(m, m.get_null_context(2))
            theirs = run_scheduler(sch, wrapped, a)
            err = (ours - theirs).abs().max().item()
            assert err < 1e-4, f"analytic euler misaligned (γ={gamma}, N={n}): {err}"
            if gamma == 0.0:
                closed = a + d.view(1, 1, 1, 1)
                assert (ours - closed).abs().max().item() < 1e-5, "euler γ=0 should hit closed form"
                assert (theirs - closed).abs().max().item() < 1e-5, "diffusers γ=0 should hit closed form"
    print("  [ok] Euler 解析场对齐: γ=0/0.4 × N=8/13 全部 <1e-4, 闭式解双验证")


def test_euler_real_unet_alignment():
    """真实 UNet + C 条件全链路对齐 (N=8, 含 ImageContextEncoder)。
    验证完整约定链: timestep 缩放 -> 时间换算 -> 速度符号 -> Euler 更新。
    N=8 时 sigma=i/8 均为 float32 精确值, 两边 timestep 嵌入逐位一致。"""
    m = tiny_model(solver='euler', sample_step=8)
    m.eval()
    a, c_img = rand_image(2, 32), rand_image(2, 32)
    ctx = m.get_cond_stage_context(c_img, 2)  # eval 模式: 无 dropout, 确定性 tokens

    ours = m.sample(a, c=c_img, clip_denoised=False)

    sch = make_scheduler(8)
    wrapped = DiffusersWrapper(m, ctx)
    theirs = run_scheduler(sch, wrapped, a)

    err = (ours - theirs).abs().max().item()
    rel = err / ours.abs().max().item()
    assert err < 1e-4, f"real UNet misaligned: abs={err}, rel={rel}"
    print(f"  [ok] 真实 UNet 全链路对齐 (N=8, 带 C 条件): max|Δ|={err:.2e} (rel {rel:.2e})")


def main():
    if not HAS_DIFFUSERS:
        print("SKIP: diffusers 未安装 (pip install diffusers --no-deps 后重跑)")
        return 0
    import diffusers
    print(f"diffusers {diffusers.__version__}")
    torch.manual_seed(0)
    tests = [
        test_fd_velocity_in_sigma_space,
        test_sigma_grid_and_timestep_convention,
        test_euler_analytic_alignment,
        test_euler_real_unet_alignment,
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
        return 1
    print(f"ALL {len(tests)} ALIGNMENT TESTS PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
