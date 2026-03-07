# Multimodal Embedding

基于 `Qwen3-VL-Embedding-2B` 和 `ms-swift` 框架，提供完整的图文检索 Embedding 模型训练（LoRA微调）与推理验证流程。

## 1. 环境准备

确保已安装 `ms-swift` 及相关依赖。

## 2. 数据准备

本项目使用本地的 Conceptual Captions (`.parquet`) 数据，并自动下载图片构建多模态数据集。

运行以下命令：
```bash
bash download.sh
python prepare_data.py
```

## 3. 模型训练

**运行命令**:
```bash
./train.sh
```

## 4. 推理验证

**运行命令**:
```bash
python eval.py
```

## 目录结构
```
.
├── dataset/                # 生成的数据目录
│   ├── images/             # 下载的图片
│   └── train.jsonl         # 训练数据索引
├── datasets/               # 原始数据 (parquet)
├── models/                 # 本地模型文件
├── output/                 # 训练输出 (Checkpoints)
├── prepare_data.py         # 数据准备脚本
├── train.sh                # 训练启动脚本
├── eval.py                 # 评测脚本
└── README.md               # 本文档
```
