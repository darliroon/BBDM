# CBBDM 第一次跑通指南（零基础版）

> 任务：给定源图 A 与条件图 C，生成目标图 B。A→B 是近域变换（如断裂车道线→完美车道线），C 是与 A/B 域差异较大的参考图（如卫片），经 cross-attention 注入，不与 A 叠加。
> 适用：想从零准备好数据并跑通训练/测试的任何人。

## 0. 你需要准备什么

| 需要的东西 | 要求 | 说明 |
|---|---|---|
| 三组配对图片 | 每组 N 张，同尺寸（默认 256×256），RGB，**三组文件同名一一对应** | A=源图（待修复/待转换）；B=目标图（真值）；C=条件图（参考信息） |
| 一台有显卡的机器 | 24G 显存可跑（4×3090 更快） | CPU 也能跑但极慢 |
| python 环境 | 见第 2 步 | 依赖 torch / opencv / tensorboard 等 |

**对图片格式的通用要求**：

- 三组文件**同名**：`A/0001.png`、`B/0001.png`、`C/0001.png` 属于同一个样本
- A 和 B 必须**像素级对齐**（同一位置的地物、无几何偏移）——这是布朗桥建模近域变换的前提
- C 与 A/B 同尺寸、同坐标系（模型会按空间位置让 UNet 查询 C）
- 支持的图片格式：png / jpg（`datasets/utils.py` 的 `get_image_paths_from_dir` 会按扩展名检索并排序）

## 1. 组织数据目录

目录结构是**模型约定**，必须长这样（`dataset_path` 换成你的实际路径）：

```
dataset_path/
├── train/
│   ├── A/    # 源图（桥的起点）
│   ├── B/    # 目标图（桥的终点，真值）
│   └── C/    # 条件图（cross-attention 注入）
├── val/
│   ├── A/
│   ├── B/
│   └── C/
└── test/
    ├── A/
    ├── B/
    └── C/
```

**关于数据划分的重要说明**：

1. **划分在数据准备阶段完成，程序里没有随机划分**。文件放进 `train/` 就是训练集，放进 `val/` 就是验证集。配置文件里 `data.train / data.val` 段只控制 batch_size 和 shuffle，与划分无关
2. **不要随机打散划分**。如果你的样本存在空间/来源上的相似性（如同一道路的相邻切片、同一段视频的相邻帧），随机划分会导致相邻样本分属训练/测试集，指标虚高。**按来源分组划分**（道路编号、场景 ID 等）
3. 常见比例：8:1:1 或 9:0.5:0.5

**用符号链接摆放数据（推荐，省磁盘）**：

```bash
# 假设原始数据在 /data/raw/{A,B,C} 各 10000 张，按来源分组划分后：
mkdir -p /data/liufeng/my-dataset/{train,val,test}/{A,B,C}

# 把划分结果链接进去（逐文件或用脚本循环，三组都要做）：
ln -s /data/raw/A/0001.png /data/liufeng/my-dataset/train/A/0001.png
ln -s /data/raw/B/0001.png /data/liufeng/my-dataset/train/B/0001.png
ln -s /data/raw/C/0001.png /data/liufeng/my-dataset/train/C/0001.png
```

**快速自检**（数据集代码会自动断言，但提前查省得训练时才报错）：

```bash
for s in train val test; do
  echo "$s: A=$(ls dataset_path/$s/A | wc -l) \
B=$(ls dataset_path/$s/B | wc -l) C=$(ls dataset_path/$s/C | wc -l)"
done
# 三个数字应分别相等（每个 stage 内 A/B/C 数量一致、文件名一致）
```

## 2. 准备环境

**方案一：用现成环境（本机已配好）**：

```bash
# 依赖已补齐，直接用：
/home/liufeng/conda_envs/diffsynth/bin/python
```

**方案二：新机器自建**：

```bash
conda env create -f environment.yml   # 原仓库环境（较老）
# 或手动装核心依赖（torch 2.x 可用）：
pip install torch opencv-python-headless tensorboard torchsummary \
    pytorch_lightning einops more_itertools omegaconf pyyaml tqdm
```

> 注意：仓库代码在 torch 2.13 下已做兼容（修复了 `ReduceLROnPlateau(verbose=)` 移除问题、`datasets` 包名遮蔽问题），无需降级 torch。

## 3. 写配置文件

复制主模板：

```bash
cd /path/to/BBDM
cp configs/Template-CBBDM.yaml configs/my_task.yaml
```

编辑 `configs/my_task.yaml`，**最小改动只需一处**：

```yaml
data:
  dataset_name: 'my_task'                  # 结果目录名（顺手改掉）
  dataset_config:
    dataset_path: '/data/liufeng/my-dataset'   # ← 改成你的数据根目录
    image_size: 256                         # 与你的图片尺寸一致
```

其他参数保持默认即可跑通。常见可调项：

| 参数 | 默认 | 说明 |
|---|---|---|
| `data.train.batch_size` | 2 | 24G 显存建议 2-4 起步，OOM 则降 1 |
| `model.BB.params.cond_dropout` | 0.1 | 条件丢弃比例（CFG 训练），保持默认 |
| `model.BB.params.sample_step` | 200 | 采样步数，越大越慢、质量略升 |
| `training.n_epochs` | 300 | 训练轮数 |

## 4. 启动训练

```bash
cd /path/to/BBDM

# 单卡：
python main.py --config configs/my_task.yaml --train --sample_at_start --save_top --gpu_ids 0

# 多卡（更快）：
python main.py --config configs/my_task.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3
```

**必须在仓库根目录下运行**（`cd /path/to/BBDM`），否则包导入会错乱。

**怎么判断训练正常**：

1. 终端打印 loss，正常情况在几百步内明显下降
2. TensorBoard 看曲线：`tensorboard --logdir results/{dataset_name}/{model_name}/`
3. **训练初期的采样结果应接近源图 A**——这是零初始化的设计使然：模型先学"A 不动"，C 的影响随训练渐进出现。如果一开始输出就是乱码才说明有问题

结果目录结构：

```
results/{dataset_name}/{model_name}/
├── image/          # 定期采样可视化（每 sample_interval epoch）
├── checkpoint/     # 模型权重（每 save_interval epoch 保存）
└── log/            # tensorboard 日志
```

## 5. 测试（生成结果）

```bash
python main.py --config configs/my_task.yaml --sample_to_eval --gpu_ids 0 \
    --resume_model results/{dataset_name}/{model_name}/checkpoint/latest_model_300.pth
```

（`latest_model_XXX.pth` 的数字换成实际训到的 epoch）

输出四个文件夹（`results/.../test_sample/` 下）：

| 文件夹 | 内容 |
|---|---|
| `condition/` | 输入 A（源图） |
| `context/` | 条件 C |
| `ground_truth/` | 真值 B |
| `200/` | 模型输出（200 = 采样步数） |

## 6. 看效果好不好

**肉眼看**：`200/xxx.png` 与 `ground_truth/xxx.png` 并排对比，关注任务关心的变化区域是否正确、有没有引入假变化、颜色/结构有没有串扰。

**量化指标（按任务选）**：

- 通用：PSNR / SSIM / LPIPS（仓库自带 `preprocess_and_evaluation.py` 和 `evaluation/`）
- FID（分布层面）：`evaluation/FID.py`
- 任务级指标：根据你的任务定义（如二值图用 per-channel IoU、修复类任务统计缺陷数量的变化）

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| import 错乱（`No module named 'xxx'`） | 不在仓库根目录运行 | `cd /path/to/BBDM` 后再运行 |
| 数据集断言报错 A/B/C 数量不一致 | 文件没放齐或文件名不同名 | 检查第 1 步的自检命令 |
| OOM 显存不足 | batch 太大 | `batch_size` 降到 1 |
| 输出全是纯色/乱码 | 未收敛或配置改坏 | 继续训；对照模板检查改动 |
| 想调 C 的影响力 | — | 配置 `testing.guidance_scale`（如 1.5），>1 启用 CFG 放大 C 的影响（需训练时 `cond_dropout>0`，默认 0.1 已开） |
| 断点续训 | — | 加 `--resume_model <ckpt> --resume_optim <optim_ckpt>` |

## 8. 想进一步折腾时

- **C 到底有没有用？**：对比 `sample(A, C)` 与 `sample(A, ∅)`（C 置空）——输出几乎一样说明模型忽略了 C，可增大 `cond_dropout` 或加深 `transformer_depth`
- **对照实验**：`condition_key: "nocond"` + 二元组数据集（`custom_aligned`，只用 A/B）= 纯 BBDM baseline，报告必备对照
- **潜空间版**：`configs/Template-CBBDM-Latent-f4.yaml`（更快但需先验证 VQGAN 对你的图像类型的重建质量——细线/二值图慎用）
