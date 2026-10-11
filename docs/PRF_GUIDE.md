# PRF (Paired Rectified Flow) 使用指南

> PRF 是 CBBDM 的**可替换内核**：A→B 变换从"布朗桥扩散"换成"配对整流流"，C 条件注入机制、数据管线、UNet 结构完全不变。本仓库现为**双内核架构**，通过配置文件一个字段切换。
> 设计思想对照详见 [PRF_vs_CBBDM.md](PRF_vs_CBBDM.md)。

## 1. 内核切换（组件化）

四种内核共用 `BBDMRunner`、`custom_triple` 数据集、`CondStageParams` 与 UNet 结构，仅 `model.model_type` 一个字段决定加载哪个内核：

| model_type | 内核 | 配置模板 | 空间 |
|---|---|---|---|
| `BBDM` | CBBDM（布朗桥扩散） | `configs/Template-CBBDM.yaml` | 像素 |
| `LBBDM` | CBBDM 潜空间版 | `configs/Template-CBBDM-Latent-f4.yaml` | VQGAN f4 |
| `PRF` | **PRF（整流流）** | `configs/Template-PRF.yaml` | 像素 |
| `LPRF` | **PRF 潜空间版** | `configs/Template-PRF-Latent-f4.yaml` | VQGAN f4 |

四种内核参数量完全一致（同 UNet + 同 ImageContextEncoder），训练超参（lr / EMA / accumulate_grad_batches / batch_size）模板默认值也一致——**改一个字段即为同条件对比**。

新增文件（原有代码零改动，Runner 仅新增两个 elif 分支）：

```
model/BrownianBridge/PairedRectifiedFlowModel.py       # PRF 内核 (像素)
model/BrownianBridge/LatentPairedRectifiedFlowModel.py # PRF 内核 (潜空间)
configs/Template-PRF.yaml / Template-PRF-Latent-f4.yaml
shell/PRF.sh
tests/test_prf.py                                       # 测试套件 (14 项)
tests/test_prf_diffusers_alignment.py                   # diffusers 数值对齐 (4 项)
tests/bench_inference.py                                # 推理速度基准
```

## 2. 数据准备

与 CBBDM 完全相同：`dataset_path/{train,val,test}/{A,B,C}/同名.png`，详见 [first_run.md](first_run.md) 第 1 节。**已为 CBBDM 准备的数据集无需任何改动，直接复用。**

## 3. 训练

```bash
cd /path/to/BBDM
cp configs/Template-PRF.yaml configs/my_prf.yaml
# 编辑 my_prf.yaml: dataset_path 改为实际路径 (唯一必改项)

# 单卡
python main.py --config configs/my_prf.yaml --train --sample_at_start --save_top --gpu_ids 0
# 多卡
python main.py --config configs/my_prf.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3
```

结果目录：`results/{dataset_name}/PRF-pixel/{image,checkpoint,log}`，TensorBoard 查看同 CBBDM。

**训练正常判据**：初期采样结果接近源图 A（C 编码器零初始化，C 的影响随训练渐进出现——与 CBBDM 同款语义）。

## 4. 测试与评估

```bash
python main.py --config configs/my_prf.yaml --sample_to_eval --gpu_ids 0 \
    --resume_model results/lane_lines/PRF-pixel/checkpoint/latest_model_XXX.pth
```

输出目录同 CBBDM：`condition/`（A）、`context/`（C）、`ground_truth/`（B）、`25/`（结果，**25 = sample_step**）。评估命令同 CBBDM（`preprocess_and_evaluation.py` / `fidelity`），只需把结果目录名从 `200` 换成 `25`。

## 5. PRF 专属参数（`model.BB.params` 下）

| 参数 | 默认 | 说明 |
|---|---|---|
| `gamma` | `0.0` | **插值噪声强度 γ**。`0` = 纯 rectified flow（直线插值，推荐起点）；`>0` = 路径加 `γ·t(1-t)·z` 噪声正则（起点终点仍严格为 A/B），建议消融 `0 / 0.05 / 0.1` |
| `t_sampler` | `'lognorm'` | 训练时间步分布。`lognorm`（SD3 默认，中间时段加权）或 `uniform` |
| `lognorm_mean` / `lognorm_std` | `0.0` / `1.0` | logit-normal 的 loc / scale |
| `solver` | `'heun'` | 采样 ODE 求解器。`heun`（2 阶，每步 2 次前向，推荐）或 `euler`（1 阶，每步 1 次前向） |
| `sample_step` | `25` | ODE 步数。heun×25（50 NFE）≈ CBBDM 200 步质量，快约 4 倍 |
| `loss_type` | `'l1'` | 速度回归损失（同 CBBDM 语义） |
| `cond_dropout` | `0.1` | C 条件丢弃（CFG 训练），与 CBBDM 同值 |
| `num_timesteps` | `1000` | timestep 嵌入尺度（t×1000），**勿改** |

CBBDM 专属参数（`mt_type / objective / eta / max_var / skip_sample / sample_type`）在 PRF 下被忽略，配置里保留不报错——便于两份模板互相对照。

**推理期 C 影响力调节**：`testing.guidance_scale`（如 1.5），与 CBBDM 完全相同的 CFG 公式 `v = (1+w)·v(C) − w·v(∅)`。

## 6. 与 CBBDM 同条件对比实验建议

```bash
# 相同数据、相同 batch/lr/EMA/epochs, 唯一差异 = model_type
CONFIG_PRF=configs/my_prf.yaml    # model_type: PRF
CONFIG_CBBDM=configs/my_cbbdm.yaml # model_type: BBDM (其余字段保持一致)
bash shell/PRF.sh                  # 或 bash shell/CBBDM.sh
```

- 采样步数换算：PRF `heun×25` ≈ 50 NFE vs CBBDM 200 步（每步 1 NFE）；对比时报告两者 NFE 与墙钟时间
- γ 消融是 PRF 侧最重要的实验：`γ=0`（纯直线）vs `γ=0.05/0.1`（带噪声正则）观察 OOD 鲁棒性差异
- C 有效性检验同 CBBDM：对比 `sample(A, C)` / `sample(A, ∅)`（`c=None` 采样已支持）

## 7. 测试套件

```bash
cd /path/to/BBDM && python tests/test_prf.py                       # 14 项 (CPU)
cd /path/to/BBDM && python tests/test_prf_diffusers_alignment.py   # 4 项, 需 diffusers (缺失自动 SKIP)
```

**14 项主套件**全过：插值/速度公式、x_t 构造、梯度流、t 分布、timestep 缩放约定、ODE 求解器解析精度（Heun 精确 / Euler 一阶收敛）、CFG 组合、cond_dropout、零初始化、采样接口、玩具任务端到端收敛（loss 0.044→0.0001，ODE 采样 MAE=0.005）、潜空间版、Runner 全链路（PRF+LPRF）、CBBDM 回归。

**4 项 diffusers 对齐**（`pip install diffusers --no-deps`）：与 `FlowMatchEulerDiscreteScheduler`（SD3 生态）数值对齐——有限差分速度验证、sigma 网格/timestep 缩放约定、解析场 Euler 对齐（γ=0/0.4）、真实 UNet 全链路逐位一致（max|Δ|=0）。

推理速度基准（GPU + checkpoint）：`python tests/bench_inference.py <gpu_id>`，详见 [推理速度优化.md](推理速度优化.md)。

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 想恢复 CBBDM | — | `model_type` 改回 `BBDM`/`LBBDM`，其余不动 |
| PRF 输出与 CBBDM 风格不同 | PRF 采样是确定性 ODE，同一 A+C 输出唯一 | 正常；多样性需求见 γ 消融与设计文档第 6 节 |
| `sample_step` 调多少 | — | heun 从 25 起调；不足则 50；再不足换 `solver: euler` + 100 |
| γ>0 后采样仍确定 | 设计如此：γ 只影响训练路径，采样恒为 ODE | 见设计文档第 4 节 |
| OOM | — | 同 CBBDM：降 batch / 用 LPRF 潜空间版 |
