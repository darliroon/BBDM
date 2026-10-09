# CBBDM 架构与数据流详解

> 配套阅读：[改进点.md](改进点.md)（改了什么）· 本文（为什么这样设计、数据怎么流转）
> 以 `configs/Template-CBBDM.yaml` 为例：`batch=2, 图=256×256×3, token_grid=32, context_dim=512`

## 1. 全景：一条数据的三条命脉

```
B ──┐                                                       
A ──┼─ q_sample ─► x_t(2,3,256,256) ─► [UNet 主干] ─► obj_hat ─► l1 loss
    │                ↑t(2,)                ↑
C ──┴─ ImageContextEncoder ─► context(2,1024,512) ─► CrossAttn k/v   (只此一路, 不 concat)
                                      
采样: A ─► x_T ─► [UNet×200步, 每步向 B 修正一点] ─► B'
```

核心记忆点：

- **A/B 只在 `q_sample` 的两行公式里出现**（决定 x_t 长什么样，A 是采样起点）
- **C 只在 CrossAttention 的 k/v 里出现**（决定去噪时"参考什么"）
- **两者唯一的交汇处是 UNet 内部的特征图**

### A 的路径：不进任何编码器，只走桥数学

训练时 UNet **从未单独见过干净的 A**。A 的信息全部溶进 `q_sample` 的混合物：

```python
x_t       = (1 - m_t)*B + m_t*A + sigma_t*noise   # UNet 的输入
objective = m_t*(A - B) + sigma_t*noise           # 学习目标 ('grad')
```

采样时 A 才直接出场：反向过程从 `x_T ≈ A` 起步（m_T≈0.999），逐步去噪落向 B。

### C 的路径：独立编码 → token 序列 → cross-attention

C 经 `ImageContextEncoder` 编成 1024 个 token，在 UNet 的 SpatialTransformer 里被 k/v 投影后查询。**与 A 零叠加**。

## 2. 生动版：车道线修复师

| 角色 | 对应 |
|---|---|
| 修复师 | UNet（去噪网络） |
| 画坏的图 | A（断裂车道线） |
| 应有的成品 | B（完美车道线） |
| 桌上摊开的卫星底图 | C（卫片） |
| 画坏的"程度" | 时间步 t |

**A 怎么进来**：老师（q_sample）把成品 B 和坏图 A 倒进一杯"鸡尾酒"，t 小酒里主要是 B，t 大主要是 A，再蒙层雾。修复师学会的是"给定任意浑浊度的酒，反推调制轨迹"。考试时从"几乎全是 A"的状态起步往回修。

**C 怎么进来**：不是垫在画底下（concat 会互相污染），而是摊在旁边。修复师修到某位置时抬眼扫一遍底图，目光在相关位置停留更久（attention 权重），读回信息后落笔。token 网格与画布同坐标系，"看一眼"天然空间对齐，但也能瞟到别处——弯道正需要这种远端感受。

**两个精妙细节**：

- **实习生条款（零初始化）**：C 通路是新来的实习生，入职第一天嘴上贴封条（proj/pos 全零），说什么都不影响画布；A→B 老手艺练稳后才逐渐解封（ControlNet 式软启动）
- **盲修考试（cond_dropout=0.1）**：10% 概率抽走底图，逼修复师练"没底图照样修"的基本功——为 CFG 留口子，也防过度依赖卫片

## 3. 伪代码：训练一步（含维度）

```python
def train_step(batch):
    x0 = B   # (2, 3, 256, 256)   # 桥终点
    y  = A   # (2, 3, 256, 256)   # 桥起点
    c  = C   # (2, 3, 256, 256)   # 条件

    t = randint(0, 1000, (2,))        # (2,) 每个样本随机一个桥上位置

    # ---------- C 的路径 ----------
    context = ImageContextEncoder(c, drop_prob=0.1)   # → (2, 1024, 512)

    # ---------- A/B 的路径: 建桥 ----------
    m_t     = m_table[t]               # (2, 1, 1, 1)
    sigma_t = sqrt(var_table[t])       # (2, 1, 1, 1)
    noise   = randn_like(x0)           # (2, 3, 256, 256)
    x_t       = (1 - m_t)*x0 + m_t*y + sigma_t*noise
    objective = m_t*(y - x0) + sigma_t*noise

    # ---------- 聚合 ----------
    objective_recon = UNet(x_t, timesteps=t, context=context)  # (2, 3, 256, 256)

    loss = |objective - objective_recon|.mean()       # l1
    # 每 8 步: EMA.update(UNet)
```

## 4. ImageContextEncoder 内部变换逐行拆解

```python
f = conv3x3(c)                # (2, 128, 256, 256)  stem: Conv+GroupNorm+SiLU
f = block(f)                  # (2, 128, 128, 128)  stride-2 ×3, 视野逐级翻倍
f = block(f)                  # (2, 128,  64,  64)
f = block(f)                  # (2, 128,  32,  32)
f = pool(f, (32, 32))         # (2, 128,  32,  32)  保险对齐 (当前配置为 no-op)

tokens = f.flatten(2).T       # (2, 1024, 128)      ★网格 → 句子, token_n = 行×32+列
tokens = Linear_零初始化(tokens) + pos_emb   # (2, 1024, 512)

keep = rand(2) >= 0.1         # (2,)  整样本判定
tokens = where(keep, tokens, null_emb)   # (2, 1024, 512) → context
```

关键变换的含义：

| 步骤 | 发生了什么 |
|---|---|
| stem conv | "颜色" → 128 种"局部图案特征"，空间尺寸不变 |
| 3× stride-2 | 分辨率 256→32，感受野从 3×3 涨到约 17×17 像素；与 UNet 下采样节奏对齐 |
| flatten+transpose | 每个空间格子变成一个 token，**token 编号 = 空间位置**（同坐标系的来源） |
| Linear(128→512) | 逐 token 独立翻译成 UNet 的语言（context_dim 与 to_k/to_v 输入严格一致） |
| pos_embedding | flatten 后空间排布只剩序号，位置嵌入把空间信息显式注入 |
| 零初始化 | 第 0 步 tokens≡0 → k=v=0 → attention 输出 0，C 完全哑音 |
| null_embedding | "无 C"被建模为可学习模式而非填 0；10% 整样本替换（广播 (2,1,1)，1024 个 token 一起换） |

## 5. context 在 UNet 内部的使用

### 传递：只算一次，广播消费

- context 在 `BrownianBridgeModel.forward` 里**计算一次**
- `UNetModel.forward` 把同一个张量**按引用**传给每个模块
- `TimestepEmbedSequential` 按类型分发：**只有 SpatialTransformer 收 context**，ResBlock/采样层完全不知道 C 的存在

按 `channel_mult=(1,2,4,8,8), attention_resolutions=(32,16), num_res_blocks=2`，消费点共 **11 处**：

| 位置 | 分辨率 | SpatialTransformer 数 |
|---|---|---|
| 编码器 level3 | 32×32 | 2 |
| 编码器 level4 | 16×16 | 2 |
| middle block | 16×16 | 1 |
| 解码器 level4 | 16×16 | 3 |
| 解码器 level3 | 32×32 | 3 |

**共享的是"原文"，不是"解读"**：11 处各自持有独立的 `to_k/to_v` 权重——同一个 context，11 种读法（浅层读布局、深层读语义），反向传播时 11 条梯度汇流回同一个编码器。

### 消费：SpatialTransformer → CrossAttention

```python
# 32×32 层, ch=1024, num_head_channels=64 → 16 头 × 64 维
x_in = x                                  # (2, 1024, 32, 32)
x = proj_in_1x1conv(norm(x))              # (2, 1024, 32, 32)
x = rearrange(x, 'b c h w -> b (h w) c')  # (2, 1024, 1024)  画布也 token 化
#   BasicTransformerBlock:
x = attn1(norm1(x))            + x   # 自注意力: 画布内部互相看
x = attn2(norm2(x), context)   + x   # ★交叉注意力: 画布查询卫片
x = ff(norm3(x))               + x   # 前馈
x = rearrange(x, 'b (h w) c -> b c h w')
x = proj_out_1x1conv(x)                  # 零初始化 (zero_module)
return x + x_in                          # 残差
```

CrossAttention 本体：

```python
q = to_q(x)         # (2, 1024位置, 1024)
k = to_k(context)   # (2, 1024token, 1024)  ← C 的"地址标签"
v = to_v(context)   # (2, 1024token, 1024)  ← C 的"实际内容"
# 多头拆分: (32, 1024, 64) 各自 (2样本×16头)
sim  = q @ k.T * scale          # (32, 1024, 1024)  [画布位置i, 卫片tokenj] 相关性
attn = softmax(sim, dim=-1)     # 每个画布位置对 1024 个卫片 token 的注意力分布
out  = attn @ v                 # 按权重把卫片内容"取回"画布位置
```

这张 `(位置×token)` 相关性表就是聚合的数学本体。16×16 层位置数变 256，但 k/v 仍来自 1024 个 token——粗粒度位置同样可查询全部卫片。

## 6. 为什么注意力选在 32/16 层

**算力硬约束**（自注意力 O(N²)）：

| 特征层 | 位置数 | 注意力矩阵 | 可行性 |
|---|---|---|---|
| 256×256 | 65,536 | ≈43 亿 | 爆显存 |
| 64×64 | 4,096 | 1700 万 | 昂贵 |
| **32×32** | **1,024** | **100 万** | 甜点区 |
| **16×16** | **256** | 6.5 万 | 廉价 |

**信息性质匹配**：A/B 同域（白底红线线图），C 异域（航拍卫片）。浅层特征是像素级的，强行融合会互相污染；C 只需贡献语义级"结构建议"（路怎么弯），颜色线条细节由桥端点和 skip 在浅层自己解决。

**任务需求**：修弯道需要远端线索，卷积浅层感受野不够；注意力全连接，深层一步可达全局。config 注释 `# 线连续性需要远端感受` 即此意。

**为什么恰好 32+16**：

- 32×32 与 C 的 `token_grid: 32` 严格一一对齐（1024 位置 ↔ 1024 token）
- 16×16 是 `channel_mult` 的最深层；再深（8×8）对 2-3px 线条太粗
- middle block 是信息瓶颈处，语义注入性价比最高

这是 SD/Imagen 一脉相承的标准实践：**浅层管"画"（像素细节），深层管"想"（结构语义）**。

## 7. EMA：影子权重

`runners/base/EMA.py` + config `model.EMA`：

```yaml
EMA:
  use_ema: True
  ema_decay: 0.995          # 平滑系数
  update_ema_interval: 8    # 每 8 step 更新
  start_ema_step: 20000     # 此前 shadow=直接复制 (with_decay=False)
```

- 训练旁路维护影子权重：`shadow = decay·shadow + (1−decay)·param`
- **验证/采样/测试自动换 EMA 权重**（apply_ema → 推理 → restore_ema），checkpoint 额外存 `ema`
- 扩散模型对权重抖动敏感，EMA 通常显著提升采样质量，所有模板默认开启

## 8. 数据增强现状（flip）

唯一的增强开关是 `data.dataset_config.flip`（所有模板默认 True）：

- 实现在 `datasets/base.py` 的 `ImagePathDataset`：**index 驱动的确定性翻倍**——数据集长度 ×2，后半段 index 永远翻转，每个 epoch 增强完全相同
- 仅 train 阶段生效（`stage == 'train'` 才开）
- 配对数据 A/B/C 同 index 驱动翻转，几何同步
- 无 RandomCrop/ColorJitter/Rotation 等

**改造为每 epoch 随机增强的要点**：决策放 `custom_triple.__getitem__` 里掷一次硬币、三方共享（不能各自挂 `RandomHorizontalFlip`，会各自掷硬币破坏配对）。PyTorch DataLoader 每 epoch 重抽 base seed，跨 epoch 自动变化，无需额外设置。

**弯道任务注意**：水平翻转会把左弯变右弯——若样本方向分布不均，flip 翻倍机制天然拉平左右弯道；这是特性不是 bug。

## 9. 类别不均衡的增量训练策略（弯道样本补充场景）

场景：V1 用 8000 张（多为直路）训练，弯道学不好；新增 5000 张弯道样本。

**数据配比**：混合训练（13000 张，弯道 ~38%），不要只用弯道微调（灾难性遗忘）；弯道仍短板再上调比例至 ~1:1。**val/test 也要加弯道并固定**，直路/弯道分开统计，否则验证 loss 无法量化弯道改善。

**增量 vs 重训**：先在 V1 上低 LR 增量，不行再全量重训。增量配置三个必改项：

```yaml
model:
  model_load_path: 'V1 checkpoint'   # 只加载模型权重+EMA
  # 不设 optim_sche_load_path → 优化器全新, 新 LR 生效
training:
  n_epochs: 450      # 必调大! global_epoch 从 checkpoint 恢复, 不调=零训练
  n_steps: 600000
model.BB.optimizer:
  lr: 2.e-5          # 降到原 1/5~1/2
```

坑：`load_model_from_checkpoint` 恢复 `global_epoch/global_step`，训练从那里继续；若同设 `optim_sche_load_path`，Adam 动量与 plateau scheduler 一并恢复，LR 不受新值控制。EMA 无需操心（step 已超 `start_ema_step`，shadow 随 checkpoint 恢复）。

**放弃增量的信号**：弯道始终学不出连续弧线，或弯道改善的同时直路明显退化——此时用混合数据从零重训。

## 10. 常见问题速查

**`model_type` 是什么？**
像素/潜空间开关（`BBDMRunner.initialize_model`）：`"BBDM"` → 像素空间 `BrownianBridgeModel`；`"LBBDM"` → 潜空间 `LatentBrownianBridgeModel`（A/B 经 VQGAN 编解码，VQGAN 冻结）。条件有无由数据集和 `condition_key` 决定，与 `model_type` 正交。

**像素空间经过 VQGAN 吗？**
不经过。`BrownianBridgeModel` 无任何 VQGAN 引用，config 也无需 `VQGAN` 段。潜空间版才有（`LatentBrownianBridgeModel` 加载 `VQModel` 并冻结）。注意：潜空间版里 C 也**不走 VQGAN**，走自己的 ImageContextEncoder。

**本仓库是 BBDM 还是 CBBDM？**
基线是 [xuekt98/BBDM](https://github.com/xuekt98/BBDM)（CVPR 2023），本仓库为其 CBBDM 改进版（条件注入机制见 [改进点.md](改进点.md)）。用哪个模板训练决定具体架构：`Template-BBDM.yaml`（无条件基线）/ `Template-CBBDM.yaml`（像素空间 CBBDM）/ Latent 系（潜空间）。
