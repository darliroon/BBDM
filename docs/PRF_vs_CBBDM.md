# PRF 与 CBBDM 设计思想逐条对照

> 问题源头：CBBDM 的设计思想是「**A→B 是邻域变换，C 是额外信息注入**」，以此规避"A+C→B 复合变换"。本文逐条论证 PRF（Paired Rectified Flow）内核是否保留并强化了这一思想，以及两内核的实质性差异与取舍依据。
>
> 实现文件：CBBDM = [BrownianBridgeModel.py](../model/BrownianBridge/BrownianBridgeModel.py)，PRF = [PairedRectifiedFlowModel.py](../model/BrownianBridge/PairedRectifiedFlowModel.py)。

## 0. 两个内核的数学一句话

| | CBBDM（布朗桥扩散） | PRF（配对整流流） |
|---|---|---|
| 前向路径 | `x_t = (1−m_t)·B + m_t·A + σ_t·ε`，t 离散 0…T | `x_t = (1−t)·A + t·B + γ·t(1−t)·z`，t 连续 0…1 |
| 学习目标 | 桥目标（grad/noise/ysubx 参数化） | 速度场 `v = (B−A) + γ(1−2t)·z` |
| 采样 | 反向随机祖先采样（200 步，每步重注入 A） | 确定性 ODE 积分 A→B（heun 25 步 / 50 NFE） |
| 理论出处 | BBDM (CVPR 2023) | Rectified Flow (ICML 2023) + Stochastic Interpolants (ICML 2023)；工程约定同 SD3/Flux |

γ=0 时 PRF 即纯 rectified flow；γ>0 是 stochastic interpolants 框架的标准噪声正则形式（**同一条路径上共享同一个 z**，这是它与扩散桥的独立噪声本质不同，也是速度回归无偏的关键）。

## 1. 逐条对照：CBBDM 的设计思想在 PRF 中的对应物

### 1.1 「生成从 A 开始，不从噪声开始」

| CBBDM | PRF |
|---|---|
| 反向采样从 `x_T ≈ A`（源图）出发（[p_sample_loop](../model/BrownianBridge/BrownianBridgeModel.py) 以 y=A 为初值） | ODE 初值 `x(0) = A` 严格成立：插值在 t=0 处 `γ·0·1·z = 0`，**逐像素精确等于 A** |

**结论：完全保留**，且更严格——CBBDM 的 x_T 是"桥端点附近加噪的分布"，PRF 的起点是 A 本身。

### 1.2 「A→B 是邻域变换」

| CBBDM | PRF |
|---|---|
| x_t 活在 A–B 线段附近（m_t 调度 + 噪声），"近域"由调度形状隐式保证 | x_t **严格位于 A–B 直线线段上**（γ=0 时） |
| 网络学桥目标（如 grad 参数化下的位移） | 网络学速度场 `v = B−A`：**A==B 的像素处 v≡0，只有断裂 gap 处 v≠0** |

**结论：保留且字面成立**。"邻域变换"四个字在 PRF 里从隐式（调度 + 噪声近似）变成显式（残差速度场）：网络容量被结构性约束为只花在 gap 区域，已正确的车道线像素速度为零、自然保持。

### 1.3 「C 是额外信息注入，不与 A 叠加」

| CBBDM | PRF |
|---|---|
| [ImageContextEncoder](../model/BrownianBridge/cond_encoder.py)：C → 1024 token → UNet cross-attention；UNet 空间输入始终只有 x_t | **同一模块、同一零初始化、同一代码路径，逐行复用**（PRF 内核调用与 CBBDM 完全相同的 `get_cond_stage_context`） |

**结论：零改动迁移。** UNet 输入永远是 `(x_t, t, context)` 三元组，从不出现 A⊕C 通道拼接。

### 1.4 「规避 A+C→B 的复合变换」

| CBBDM | PRF |
|---|---|
| 网络不需要"融合 A、C 特征后渲染 B"，只需在桥路径上逐步去噪，C 只是去噪器的辅助输入 | 网络只在线段上估计局部位移；已有线区域 v≈0（保留 A），断裂 gap 区域 v≠0 且方向证据只能来自 C 的 cross-attn 查询 |

**结论：保留，且信息分工更清晰**——"A 给已有线、C 补 gap 证据"从训练动力学中自发涌现（速度场的结构性零点），不再依赖调度形状的间接保证。

### 1.5 「cond_dropout + CFG，推理期 C 影响力可调」

| CBBDM | PRF |
|---|---|
| 训练 10% 丢 C 学 null token；推理 `ε = (1+w)·ε(C) − w·ε(∅)` | 同款 null_embedding、同款丢弃逻辑、同款组合公式 `v = (1+w)·v(C) − w·v(∅)`（速度场上做 CFG，与 SD3 一致） |

**结论：完全保留**，测试套件对 CFG 组合公式做了闭式解验证。

### 1.6 「潜空间备选、零初始化渐进引入、EMA、三元组几何同步」

全部逐行对应：LPRF 镜像 LBBDM 的 VQGAN 编解码与"C 不经 VQGAN"策略；`apply(weights_init)` 后 proj/pos 归零、C 的影响从 0 渐进长出；数据集 `custom_triple` 直接复用。

## 2. 结构总览：什么变了，什么没变

```
没变 (组件级复用):
  ImageContextEncoder (C → tokens, 零初始化, null token)
  UNet 结构与权重初始化 (参数量逐一致: 553.63M vs 553.64M)
  custom_triple 数据管线 / BBDMRunner / EMA / 优化器与调度器配置段
  CFG 公式 / cond_dropout / guidance_scale 语义

变了 (仅内核):
  训练: q_sample(桥) → interpolate(直线+γ噪声); objective(桥目标) → v_target(速度)
  时间: 离散 t∈{0..999} + m_t 调度 → 连续 t∈(0,1) + lognorm 采样 (SD3 约定)
        UNet 嵌入输入 t×1000, 小数 timestep 原生支持
  采样: 200 步随机反向过程 → N 步确定性 ODE (euler/heun)
  可调: 新增 gamma / t_sampler / solver; 旧参数 mt_type 等被忽略
```

## 3. 一个真实差异：锚定机制（唯一需要在实验中盯住的点）

这是两内核**唯一**的实质性语义差异，也是 PRF 相对 CBBDM 的已知弱点：

- **CBBDM 每步重锚定**：反向后验均值解析地重注入 A（`x_{t-1} = (1−m)·x̂₀ + m·A + 校正项`）。单步预测有误差时，轨迹每步都被拉回 A–B 流形附近。
- **PRF 只在起点锚定**：A 仅作为初始条件进入，之后 `x_{t+dt} = x_t + v̂·dt` 的轨迹完全由学到的速度场决定，没有解析项钉住轨迹。

对本任务影响有限的理由：映射近确定性、v 场结构极简单（稀疏位移）、推理从真实 A 出发时前几步即在训练分布内。paired rectified flow 的 img2img 工作（如 InstaFlow/PE-Flow 一系）也验证了这条路线。**若实验暴露 OOD 脆弱性（测试集 A 的断裂模式偏离训练分布时轨迹跑飞），γ>0 就是为此准备的旋钮**：噪声让训练路径覆盖 A 附近的邻域，速度场相应获得"误差校正"能力（I²SB 的经验：桥噪声对 img2img 鲁棒性有实际贡献）。

## 4. γ 的语义（避免误用）

- **γ 只作用于训练路径，不改变采样过程**：`γ·t(1−t)·z` 在 t=0,1 处为零，端点严格是 A/B；采样永远是无噪声 ODE。这与扩散模型"采样时逐步加噪去噪"完全不同。
- **速度目标无偏**：训练目标 `u_t = (B−A) + γ(1−2t)·z` 恰为路径全导数 ẋ_t，其条件期望等于 ODE 速度场（stochastic interpolants, Albergo et al. ICML 2023 的核心定理）。**前提是整条路径共享同一个 z**——本实现如此。若沿用扩散桥的噪声调度（σ_t ∝ √(t(1−t))，按 √t 消失）并对其做速度回归，形式目标 u_t = (B−A) + σ_t'·ε 的方差在端点无界（σ_t' ~ 1/√(t(1−t))），且对应速度场在 t=0,1 处奇异（方差膨胀/压缩需要 O(1/√t) 的无界速度，特征线不唯一，ODE 积分器从起点出发走平凡特征线）——即"naive 桥路径速度回归"的病态。γ·t(1−t)（按 t² 消失）正是为规避它而设：目标方差有界、速度场正则、ODE 传输定理成立。
- **γ 与多样性的关系**：γ 不会让采样"多解"（ODE 确定）。多样性需要随机采样器（stochastic interpolant 的 reverse SDE），当前未实现，因为本任务（车道线补全）是近确定性映射，不需要。

## 5. 速度：为什么默认 heun×25

| | CBBDM | PRF euler | PRF heun |
|---|---|---|---|
| 采样步数 | 200 | 50 | 25 |
| 每步前向次数 | 1（CFG 时 2） | 1（CFG 时 2） | 2（CFG 时 4） |
| 总 NFE（无 CFG） | 200 | 50 | 50 |
| 阶数 | —（随机过程） | 1 | 2 |

heun（改进 Euler / 显式梯形）对光滑速度场误差 O(dt²)，SD3/Flux 生态的同款选择（diffusers `FlowMatchHeunDiscreteScheduler`）。对稀疏位移场任务 50 NFE 通常足够；不足则加倍步数。

## 6. 实验路线建议

1. **主对比**：CBBDM (200 步) vs PRF (γ=0, heun×25)——同数据同预算，看质量 / FID / LPIPS / 墙钟时间
2. **γ 消融**：0 / 0.05 / 0.1——重点观察测试集 OOD 样本（断裂模式罕见者）的失败率变化
3. **C 有效性**（两内核同做）：`sample(A, C)` vs `sample(A, ∅)` vs `sample(A, C_shuffled)`，输出差异即 C 贡献；差异过小说明模型忽略 C（对策：加大 `cond_dropout` 或 `transformer_depth`）
4. **步数-质量曲线**：PRF 10/25/50/100 步，与 CBBDM 50/100/200 步对照，报告质量饱和点与 NFE

## 7. 验证状态（截至本次实现）

- `tests/test_prf.py` 14 项全过：公式/梯度流/ODE 解析精度（heun 闭式解误差 <1e-4、euler 一阶收敛 0.02→0.01）/CFG 闭式解/timestep 缩放端点 {0,500,1000}/玩具任务收敛（loss 0.044→0.0001，ODE 采样 MAE=0.005）/潜空间版/Runner 全链路（PRF+LPRF 路由、训练步、采样、sample_to_eval 四目录）/**CBBDM 回归**
- 生产配置 GPU 全链路：`Template-PRF.yaml`（553.63M 参数）与 `Template-CBBDM.yaml`（553.64M）分别完成 训练→checkpoint→sample_to_eval，输出结构一致
- 参数量差异 0.01M = CBBDM 的桥调度 buffer（m_t/variance_t 等 5×1000 浮点），模型本体逐参数一致
