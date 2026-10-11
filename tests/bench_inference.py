"""CBBDM/PRF 推理速度基准: 量化各加速手段的真实 ms/NFE 与端到端墙钟时间。

用法: cd /path/to/BBDM && python tests/bench_inference.py [gpu_id]
不依赖训练质量 (用冒烟 checkpoint, 只测速度)。
"""
import sys
import os
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import dict2namespace
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.PairedRectifiedFlowModel import PairedRectifiedFlowModel

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEV = f"cuda:{GPU}"
CKPT = {
    "CBBDM": "results/lane_lines/CBBDM-pixel/checkpoint/latest_model_1.pth",
    "PRF": "results/lane_lines/PRF-pixel/checkpoint/latest_model_1.pth",
}
CFG = {}


def _make_cfgs():
    """从模板自动生成基准配置 (仅模型构造需要, dataset_path 无关紧要)"""
    templates = {"CBBDM": "configs/Template-CBBDM.yaml", "PRF": "configs/Template-PRF.yaml"}
    for kind, tpl in templates.items():
        out = f"/tmp/{kind}-bench.yaml"
        with open(tpl) as f, open(out, "w") as g:
            g.write(f.read().replace("dataset_path: 'dataset_path'", "dataset_path: 'unused'"))
        CFG[kind] = out


_make_cfgs()


def load(kind):
    with open(CFG[kind]) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader)).model
    cls = BrownianBridgeModel if kind == "CBBDM" else PairedRectifiedFlowModel
    net = cls(cfg).to(DEV).eval()
    sd = torch.load(CKPT[kind], map_location="cpu")["model"]
    net.load_state_dict(sd)
    return net


def bench_nfe(fn, x, t, ctx, iters=30, dtype=None, warmup=8):
    def once():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            fn(x, timesteps=t, context=ctx)

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        once()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    torch.backends.cudnn.benchmark = False
    print(f"GPU: {torch.cuda.get_device_name(GPU)}")

    net = load("CBBDM")
    x = torch.randn(1, 3, 256, 256, device=DEV)
    t = torch.tensor([500.0], device=DEV)
    ctx = torch.randn(1, 1024, 512, device=DEV)
    fn = net.denoise_fn

    results = {}

    # 1) fp32 基线 (仓库当前状态)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results["fp32 (现状)"] = bench_nfe(fn, x, t, ctx)

    # 2) +TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results["+TF32"] = bench_nfe(fn, x, t, ctx)

    # 3) +bf16 autocast
    results["+TF32+bf16"] = bench_nfe(fn, x, t, ctx, dtype=torch.bfloat16)

    # 4) +channels_last
    fn_cl = fn.to(memory_format=torch.channels_last)
    x_cl = x.contiguous(memory_format=torch.channels_last)
    results["+TF32+bf16+clast"] = bench_nfe(fn_cl, x_cl, t, ctx, dtype=torch.bfloat16)

    # 5) +torch.compile (bf16)
    try:
        fn_comp = torch.compile(fn_cl)
        results["+TF32+bf16+clast+compile"] = bench_nfe(fn_comp, x_cl, t, ctx,
                                                        dtype=torch.bfloat16, warmup=12)
    except Exception as e:
        print("compile failed:", e)

    print("\n=== 单次 UNet 前向 (ms/NFE, batch=1, 256²) ===")
    for k, v in results.items():
        print(f"  {k:28s} {v:7.1f} ms")

    base = results["fp32 (现状)"]
    print("\n=== 端到端投影 (按各手段 ms/NFE × 步数) ===")
    print(f"  {'配置':34s} {'NFE':>4s} {'投影耗时':>8s}")
    for label, nfe in [("CBBDM 200 步 (现状)", 200), ("CBBDM 50 步", 50), ("CBBDM 25 步", 25)]:
        print(f"  {label:34s} {nfe:4d} {base*nfe/1000:7.1f}s (fp32)")
        for k, v in results.items():
            if k == "fp32 (现状)":
                continue
            print(f"  {label+' ['+k+']':34s} {nfe:4d} {v*nfe/1000:7.2f}s")

    # 6) PRF 端到端实测 (bf16+compile 的 UNet), euler/heun 少步数
    print("\n=== PRF 端到端实测 (bf16+TF32, compiled UNet) ===")
    prf = load("PRF")
    prf.denoise_fn = torch.compile(prf.denoise_fn.to(memory_format=torch.channels_last))
    a = torch.randn(1, 3, 256, 256, device=DEV)
    c_img = torch.randn(1, 3, 256, 256, device=DEV)

    @torch.no_grad()
    def timed_sample(solver, steps, dtype=torch.bfloat16):
        prf.solver = solver
        prf.sample_step = steps
        for _ in range(2):  # warmup (含 compile)
            with torch.autocast("cuda", dtype=dtype):
                prf.sample(a, c=c_img, clip_denoised=False)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            with torch.autocast("cuda", dtype=dtype):
                prf.sample(a, c=c_img, clip_denoised=False)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    for solver, steps in [("euler", 20), ("euler", 10), ("euler", 5), ("heun", 8), ("heun", 4)]:
        dt = timed_sample(solver, steps)
        nfe = steps * (2 if solver == "heun" else 1)
        print(f"  PRF {solver}×{steps:2d} (NFE={nfe:2d}): {dt*1000:6.0f} ms/图")

    # 7) CBBDM 端到端实测对照 (bf16+compile)
    print("\n=== CBBDM 端到端实测 (bf16+TF32, compiled UNet) ===")
    cbbdm = net
    cbbdm.denoise_fn = torch.compile(fn_cl)
    b = torch.randn(1, 3, 256, 256, device=DEV)

    @torch.no_grad()
    def timed_sample_cbbdm(steps):
        cbbdm.num_timesteps = 1000
        cbbdm.linear_timesteps = torch.linspace(0, 999, steps).long().to(DEV)
        for _ in range(1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cbbdm.sample(b, c=c_img)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 3
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cbbdm.sample(b, c=c_img)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    for steps in (200, 50, 25):
        dt = timed_sample_cbbdm(steps)
        print(f"  CBBDM {steps} 步 (NFE={steps}): {dt*1000:6.0f} ms/图")


if __name__ == "__main__":
    main()
