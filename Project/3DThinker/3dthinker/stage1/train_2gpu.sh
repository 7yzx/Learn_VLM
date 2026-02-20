#!/bin/bash
# ============================================
# 双卡 DDP 训练脚本 (2x A100 80GB)
# ============================================
#
# 加速原理：
# 1. DDP (DistributedDataParallel)：两张卡各自处理不同数据，throughput 翻倍
# 2. Feature Cache：VGGT 特征缓存到内存，避免 NAS 随机 I/O
# 3. Async Prefetch：后台线程预取后续样本特征，I/O 和计算 overlap
# 4. fp16 缓存：内存中用 fp16 存储特征，占用减半
#
# 使用方式：
#   bash 3dthinker/stage1/train_2gpu.sh
#

CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    3dthinker/stage1/src/main.py \
    --model /mnt/sevenT/zixiaoy/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
    --epochs 10 \
    --task mindcube \
    --latent_size 12 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --logging_steps 20 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --stage stage1 \
    --data_path /mnt/sevenT/zixiaoy/code/Learn_VLM/Project/3DThinker/data/example.jsonl \
    --log_file ./log.txt \
    --save_model_path ./models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12_2gpu \
    --wandb_name 3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12_2gpu \
    --feature_dir /mnt/sevenT/zixiaoy/code/Learn_VLM/Project/3DThinker/data/feature_vggt/ \
    --feature_cache_fp16 \
    --num_prefetch_workers 4 \
    --num_chunks 2
