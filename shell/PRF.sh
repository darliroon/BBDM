# PRF (Paired Rectified Flow): A(断裂车道线) -> B(完美车道线), C(卫片) 为条件
# 与 CBBDM 同数据/同 UNet/同条件注入, 仅内核不同, 用于同条件对比
# 先复制 configs/Template-PRF.yaml 并填好 dataset_path, 再运行

CONFIG=${CONFIG:-configs/Template-PRF.yaml}
GPUS=${GPUS:-0,1,2,3}

# train (4x3090 DDP)
python3 main.py --config $CONFIG --train --sample_at_start --save_top --gpu_ids $GPUS

# test: 对整个 test 集采样并保存 condition/context/ground_truth/result
# python3 main.py --config $CONFIG --sample_to_eval --gpu_ids 0 --resume_model path/to/model_ckpt

# 断点续训:
# python3 main.py --config $CONFIG --train --sample_at_start --save_top --gpu_ids $GPUS \
#   --resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

# γ 消融 (修改 configs/Template-PRF.yaml 的 model.BB.params.gamma: 0 / 0.05 / 0.1)
# solver 消融: 'heun' (推荐, 2 阶) / 'euler' (更快); 采样步数 sample_step: 25 heun / 50 euler

# 评估 (在 sample_to_eval 输出目录上, 与 CBBDM 相同):
# python3 preprocess_and_evaluation.py -f LPIPS -s results/.../test/condition -t results/.../test/ground_truth -n 1
# fidelity --gpu 0 --fid --input1 results/.../test/25 --input2 results/.../test/ground_truth
