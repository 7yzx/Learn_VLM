# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

This is a multimodal RL agent for ChartQA using the VeRL (Veritable Reinforcement Learning) framework. The project trains vision-language models (VLMs) to solve chart-based QA tasks by:
1. Performing visual reasoning with "Think with Images" paradigm
2. Using image-focused tools for local attention (focus_on_rows/columns with highlight/mask/draw)
3. Applying PPO/GRPO reinforcement learning with DAPO-style dual-clip objective
4. Leveraging two-stage rollouts with tool execution

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Preparation
```bash
cd data_utils
python download_dataset.py    # Download ChartQA images and vcot annotations
# Unzip images in the appropriate directory
bash preprocess_data.sh      # Convert JSONL to parquet format
```

### Training
```bash
# Single node with 4 GPUs
bash train.sh

# Or run directly with Python
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=datasets/train_full.parquet \
    data.val_files=datasets/val_full.parquet \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.n_gpus_per_node=4
```

### Key Training Parameters (in train.sh)
- `CUDA_VISIBLE_DEVICES`: GPU assignment
- `worker.actor.global_batch_size`: Training batch size
- `worker.rollout.tensor_parallel_size`: vLLM tensor parallelism (typically 2 per node)
- `worker.rollout.n`: Number of samples per prompt for GRPO
- `worker.reward.reward_type`: Reward computation mode (batch/sequential/llm_batch/llm_double)

## High-Level Architecture

### VeRL Training Framework

The training pipeline is built around Ray + FSDP + vLLM:

1. **Ray Controller** (`verl/trainer/main.py`):
   - Entry point that creates a remote Runner
   - Initializes tokenizer/processor and creates data loaders
   - Sets up Ray worker groups for Actor, Critic, and RefPolicy roles

2. **RayPPOTrainer** (`verl/trainer/ray_trainer.py`):
   - Main training orchestrator (`fit()` method)
   - Handles worker initialization, rollouts, and updates
   - Manages checkpointing and validation

3. **Workers** (`verl/workers/`):
   - `FSDPWorker`: Container for actor/critic/ref models with FSDP sharding
   - `actor/dp_actor.py`: Policy model for generation and logprob computation
   - `critic/dp_critic.py`: Value function model (optional for GRPO)
   - `rollout/vllm_rollout_spmd.py`: vLLM-based fast sampling engine

4. **Core Algorithms** (`verl/trainer/core_algos.py`):
   - GAE, GRPO, RLOO, REINFORCE++, REMAX advantage estimators
   - PPO clipped objective with dual clip (DAPO-style)
   - KL penalty handling

### Two-Stage Rollout with Tool Use

The key innovation is the "Think with Images" mechanism:

1. **First Rollout**: Model generates initial response which may include Python code blocks containing tool calls
2. **Action Parsing** (`verl/tooluse/parse.py`): `Parser.parse()` extracts code from response
3. **Tool Execution** (`verl/tooluse/execution.py`): `CodeExecutor` runs the code in Jupyter environment with access to `focus_on_*` functions from `verl/tooluse/tools.py`
4. **Second Rollout** (optional): If tool execution succeeds, a second rollout generates the final answer based on both original and edited images

See `verl/trainer/ray_trainer.py:get_second_rollout_batch()` for the implementation.

### Data Flow

1. **Raw Data** (`data_utils/preprocess.py`):
   - Reads `chartqa_vcot/{train,val}.jsonl`
   - Builds bbox dictionaries from x/y values
   - Writes parquet files with metadata, images, prompts, and answers

2. **Dataset** (`verl/utils/dataset.py`):
   - `RLHFDataset` loads parquet data
   - Applies Jinja template from `examples/format_prompt/chartQA.jinja` to format prompts
   - Preprocesses images (resize to max_pixels/min_pixels constraints)
   - Uses HF processor for multimodal tokenization

3. **DataLoader** (`verl/trainer/data_loader.py`):
   - `create_dataloader()` instantiates StatefulDataLoader with custom collate function
   - Handles prompt/response image splitting and padding

### Reward System

Reward functions are pluggable via Python path notation:

- `examples/reward_function/refocus.py`: Rule-based scorer for ChartQA
  - Parses `FINAL ANSWER:` from response
  - Handles multiple answers with `||` separator
  - Numerical similarity scoring for numeric questions
  - Exact match for categorical answers

- `examples/reward_function/refocus_llm.py`: LLM-based judge
  - Uses external LLM (e.g., via OpenRouter) to evaluate responses
  - Returns binary scores (`<|YES|>` or `<|NO|>`)

Reward type determines execution mode:
- `batch`: Compute all rewards in one call
- `sequential`: Compute one at a time
- `llm_batch`: Batch LLM API calls
- `llm_double`: Two-stage LLM evaluation

## Configuration System

Configuration is hierarchical using OmegaConf:

1. **Default Config** (`verl/trainer/config.py`): Dataclass definitions for all parameters
2. **YAML Override** (`examples/config.yaml`): Experiment-specific settings
3. **CLI Override**: Command line arguments override YAML

Key config sections:
- `data`: Dataset paths, prompt format, length limits
- `algorithm`: Advantage estimator, KL penalty, online filtering
- `worker.actor`: Model path, optimizer, FSDP/offload settings
- `worker.rollout`: Sampling parameters, vLLM settings
- `worker.reward`: Reward function path and type
- `trainer`: Training epochs, checkpointing, logging

## Key Implementation Notes

### Bounding Box Metadata
The preprocessing step (`data_utils/preprocess.py`) extracts bbox information for chart elements and stores it in JSON metadata. This is crucial for tool functionality - tools receive bbox dicts to know where to focus/mask/draw.

### Position IDs for Qwen2-VL
The dataset uses `get_rope_index()` from `verl/models/transformers/qwen2_vl.py` to compute multi-dimensional rope positions for vision-language models. This is model-specific and automatically applied.

### FSDP Sharding Strategy
The `offload` config controls memory:
- `offload_params: true` + `offload_optimizer: true`: Maximize CPU RAM usage, minimize GPU
- `offload_params: false` + `offload_optimizer: false`: Maximize GPU usage

### Gradient Accumulation
- `global_batch_size`: Total batch size across all GPUs
- `micro_batch_size_per_device_for_update`: Gradient accumulation steps
- `micro_batch_size_per_device_for_experience`: Batch size for vLLM rollout (different from training)

### Online Filtering
When `online_filtering: true`, samples with rewards below `filter_low` or above `filter_high` percentiles are excluded from updates. This prevents outlier samples from skewing the policy.

## Directory Structure

```
├── verl/                    # Core VeRL framework
│   ├── trainer/             # Training orchestration
│   │   ├── main.py         # Entry point
│   │   ├── ray_trainer.py  # Main trainer (2000+ lines)
│   │   ├── core_algos.py   # PPO/GRPO implementations
│   │   └── config.py       # Config dataclasses
│   ├── workers/            # Ray worker implementations
│   │   ├── fsdp_workers.py # FSDP worker container
│   │   ├── actor/          # Policy workers
│   │   ├── critic/         # Value workers
│   │   ├── rollout/        # vLLM rollout workers
│   │   └── reward/         # Reward managers
│   ├── utils/              # Utilities
│   │   ├── dataset.py      # RLHFDataset
│   │   ├── tokenizer.py    # Tokenizer/processor helpers
│   │   └── fsdp_utils.py  # FSDP utilities
│   └── tooluse/            # Tool execution infrastructure
│       ├── parse.py        # Response parser
│       ├── execution.py    # Jupyter code executor
│       └── tools.py        # Focus functions
├── data_utils/             # Data preparation
│   ├── preprocess.py        # JSONL -> parquet
│   └── download_dataset.py # Download script
├── examples/               # Experiment configs
│   ├── config.yaml         # Training config
│   ├── format_prompt/      # Jinja templates
│   └── reward_function/    # Reward functions
├── train.sh                # Training launcher
└── requirements.txt        # Dependencies
```

## Troubleshooting

### vLLM OOM
- Reduce `worker.rollout.gpu_memory_utilization`
- Increase `worker.rollout.tensor_parallel_size`
- Reduce `data.rollout_batch_size`

### FSDP OOM
- Enable `worker.actor.offload.offload_params: true`
- Enable `worker.actor.offload.offload_optimizer: true`
- Reduce `worker.actor.global_batch_size` or increase `micro_batch_size_per_device_for_update`

### Tool Execution Failures
- Check that bbox metadata is properly formatted in parquet files
- Ensure `verl/tooluse/tools.py` functions can access `columns_bbox`/`rows_bbox` dicts
- Verify Jupyter executor has correct working directory

### Reward Computation Slow
- Use `batch` reward type instead of `sequential` for rule-based rewards
- For LLM rewards, ensure batch API is supported
- Consider reducing `data.rollout_batch_size` if reward compute is bottleneck
