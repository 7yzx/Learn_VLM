import torch
import sys, os
# 获取 main.py 的目录
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 回退两层到 stage1 (main.py -> src -> stage1)
stage1_dir = os.path.dirname(current_dir)
# 拼接出 transformers 的源码根目录
transformers_root = os.path.join(stage1_dir, "transformers", "src")
sys.path.insert(0, transformers_root)

# from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
# # ✅ 修改后的写法（直接从子模块导入）
try:
    # 尝试从具体文件路径导入
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers import AutoProcessor
except ImportError:
    # 如果路径不对，打印一下 transformers 到底装在哪了
    import transformers
    import os
    print("当前 Transformers 安装路径:", os.path.dirname(transformers.__file__))
    raise

from PIL import Image
import os
import logging

from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from utils import *
from task import *
from trainer_single import CustomTrainerStage1, CustomTrainerStage2
from multimodal_projector.mmprojector import projector
from feature_cache import VGGTFeatureCache, split_dataset_into_chunks, extract_idx_from_chunk

# import wandb  # Add this line
import wandb


def setup_wandb(args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Set offline mode
        # os.environ["WANDB_MODE"] = "offline"
        
        try:
            wandb.init(
                project="3dthinker-training-single",
                name=f"{args.wandb_name}_latent{args.latent_size}",
                config={
                    "model": args.model,
                    "epochs": args.epochs,
                    "task": args.task,
                    "latent_size": args.latent_size,
                    "stage": args.stage,
                    "data_path": args.data_path,
                    "save_model_path": args.save_model_path,
                    "learning_rate": 1e-5,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
                }
            )
            print("✅ Wandb initialized in offline mode")
        except Exception as e:
            print(f"❌ Wandb offline initialization failed: {e}")
            os.environ["WANDB_DISABLED"] = "true"
        
def count_images_recursive(obj):
    """
    Recursively count the number of 'type': 'image' in data structure
    """
    count = 0
    
    if isinstance(obj, dict):
        if obj.get('type') == 'image':
            count += 1
        for value in obj.values():
            count += count_images_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            count += count_images_recursive(item)
    
    return count

seed_everything(seed=42)
args=get_args()
setup_wandb(args)

logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
# print(processor.image_processor.min_pixels)
# processor.image_processor.max_pixels = 256 * 256  # Adjust pixel upper limit
# print(f"max_pixels:{processor.image_processor.max_pixels}")

if args.stage in ['stage1']: 
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
    grad_checkpointing = True
elif args.stage in ['stage2']:
    model_path = args.load_model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = False

config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage

if args.stage in ['stage1']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.float32, cache_dir=cache_dir)
elif args.stage in ['stage2']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.float32)

# 3B:mm_hidden_size=2048
# 7B:mm_hidden_size=3584
# 72B: mm_hidden_size=5120
projector_model = projector(mlp_depth=6, mm_hidden_size=2048, hidden_size=2048, latent_size = args.latent_size, fusion_input = 391, fusion_output = 1374).to(model.device)

if args.stage in ['stage1']: model.resize_token_embeddings(len(processor.tokenizer))

latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

for param in projector_model.parameters():
    param.requires_grad = True
# optimizer = torch.optim.Adam(projector_model.parameters(), lr=2e-4)

model.projector_model = projector_model

for param in model.visual.parameters():
    param.requires_grad = False

# for name, module in model.named_modules():
#     print(f"  {name}: {type(module).__name__}")


# print all paras
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
    
print(f"learning_rate: {args.learning_rate}")

def collate_fn_stage1(examples):
    ## Replace corresponding region <output_image> -> <|latent_start|><|image_pad|><|latent_end|>
    idx_list = []
    for example in examples:
        idx_list.append(example[0]['idx'])
        del example[0]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)
    # image_inputs = [<PIL.Image.Image image mode=RGB size=308x308 at 0x7F284412DA50>, <PIL.Image.Image image mode=RGB size=308x308 at 0x7F284412D9C0>]
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    ## Only user has image token
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)
    
    # assistant_examples = remove_user_images(examples)
    # assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    # assistant_text = replace_visual_spectial_tokens(assistant_text)
    # assistant_image_inputs, _ = process_vision_info(assistant_examples)
    ## Only assistant has image token
    # assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    # batch['pixel_values_latent'] = assistant_batch['pixel_values']
    # batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0] # 151665
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0] # 151666
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0] # 151667

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    # Padding for images
    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)
    
    
    v_start = processor.tokenizer("<|vision_start|>", return_tensors="pt")["input_ids"][0] # 151652
    img_pad = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0] # 151655
    v_end = processor.tokenizer("<|vision_end|>", return_tensors="pt")["input_ids"][0] # 151653
    ## After padding, input_ids should be xxx xxx 151655 15165 15165 ... 151665 151665 151665.... corresponding to image tokens
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    batch['idx'] = idx_list

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    # Find the first occurrence of start_sequence (a series of token ids) in each row (one sample). Mask this start_sequence and all tokens before it (set to -100), these positions will not be used for loss calculation.
    # Mask all pad tokens (e.g. id=0) and img tokens (e.g. id=151655) as well (set to -100). Keep remaining tokens as is for training.
    
    # Mask everything before predict to -100
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    # In each sequence, find the position of the first <image_start_token>, then mark all positions equal to <image_token> after it as 1, other positions as 0
    
    # Mark 4 latent tokens as 1
    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask
    for i, example in enumerate(examples):
        example.insert(0, {"idx": idx_list[i]})

    return batch

def collate_fn_stage2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    
    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch


preprocess_function = task_preporcess_config[args.task]
train_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in train_dataset]


if args.stage in ['stage1']:
    CustomTrainer = CustomTrainerStage1
    collate_fn = collate_fn_stage1
else:
    CustomTrainer = CustomTrainerStage2
    collate_fn = collate_fn_stage2


# ========================================================================
# 分 Chunk 预加载训练（解决内存不够放全部 VGGT 特征的问题）
# ========================================================================
#
# 原理：
#   总共 ~450GB 的 VGGT 特征，内存放不下。
#   把 dataset 切成 num_chunks 个片段（默认 2 个，即前半/后半）。
#   每次只把一个 chunk 的 VGGT 特征加载到内存，
#   在这个 chunk 上只训 1 个 epoch（chunk 内正常 shuffle），
#   然后释放内存，加载下一个 chunk，训 1 个 epoch。
#   所有 chunk 轮一遍 = 1 个等效 epoch（全部数据各看一次）。
#   循环 args.epochs 轮 → 等效完整的 args.epochs 个 epoch。
#
#   这和正常训练行为一致：
#     正常: epoch1(全部shuffle) → epoch2(全部shuffle) → ...
#     分chunk: epoch1(chunk0_shuffle + chunk1_shuffle) → epoch2(...) → ...
#
#   唯一区别：同一 epoch 内，chunk0 的样本先被看到，chunk1 后被看到。
#   但 chunk 内部是 shuffle 的，且 epoch 间顺序不同，影响很小。
# ========================================================================

if args.stage in ['stage1']:
    # 初始化特征缓存
    feature_cache = VGGTFeatureCache(
        feature_dir=args.feature_dir,
        use_fp16=args.feature_cache_fp16,
        num_workers=args.num_prefetch_workers,
    )

    # 切分 dataset 成 chunks
    num_chunks = args.num_chunks
    chunks = split_dataset_into_chunks(train_dataset, num_chunks)
    
    total_epochs = args.epochs
    
    logging.info(f"📦 Chunked training: {num_chunks} chunks, {total_epochs} epochs")
    logging.info(f"📦 Each epoch = {num_chunks} chunk loads, total loads = {total_epochs * num_chunks}")
    for i, c in enumerate(chunks):
        logging.info(f"   Chunk {i}: {len(c)} samples")
    
    # checkpoint 路径：每个 chunk 训完后保存到这里，下个 chunk 从这里 resume
    # 这样 optimizer state、scheduler state、global_step 全部保留
    chunk_ckpt_dir = os.path.join(args.save_model_path, "chunk_checkpoint")
    resume_from = None  # 第一个 chunk 不需要 resume
    
    for epoch in range(total_epochs):
        for chunk_idx, chunk_data in enumerate(chunks):
            logging.info(f"\n{'=='*30}")
            logging.info(f"🔄 Epoch {epoch+1}/{total_epochs}, "
                         f"Chunk {chunk_idx+1}/{num_chunks} "
                         f"({len(chunk_data)} samples)")
            logging.info(f"{'=='*30}")
            
            # 1. 预加载这个 chunk 的 VGGT 特征到内存
            idx_list = extract_idx_from_chunk(chunk_data)
            feature_cache.bulk_load(idx_list)
            
            # 2. 计算这个 chunk 对应的 max_steps
            #    因为 num_train_epochs 在 resume 时会基于 global_step 判断是否跳过，
            #    所以我们用 max_steps 来精确控制每个 chunk 训练的步数
            chunk_steps = len(chunk_data) // args.per_device_train_batch_size
            
            # 3. 构建 training_args
            training_args = SFTConfig(
                output_dir=chunk_ckpt_dir,
                max_steps=chunk_steps,  # 精确控制：这个 chunk 训多少步
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps if (epoch == 0 and chunk_idx == 0) else 0,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                logging_steps=args.logging_steps,
                save_strategy="steps",
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                optim="adamw_torch_fused",
                bf16=True,
                push_to_hub=False,
                remove_unused_columns=False,
                gradient_checkpointing=grad_checkpointing,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to=["wandb"],
                logging_dir='./logs/',
                logging_strategy='steps',
                # 关键：不让 Trainer 忽略已完成的 steps（因为我们用 max_steps 精确控制）
                ignore_data_skip=True,
            )
            
            # 4. 创建 Trainer 并注入特征缓存
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=chunk_data,
                data_collator=collate_fn,
                processing_class=processor.tokenizer,
            )
            trainer.set_feature_cache(feature_cache)
            
            # 5. 训练：如果有上一个 chunk 的 checkpoint，从那里 resume
            #    resume_from_checkpoint 会恢复 optimizer state + scheduler state + global_step
            if resume_from is not None and os.path.isdir(resume_from):
                logging.info(f"📂 Resuming from checkpoint: {resume_from}")
                trainer.train(resume_from_checkpoint=resume_from)
            else:
                trainer.train()
            
            # 6. chunk 训完后保存 checkpoint（包含 model + optimizer + scheduler + global_step）
            #    Trainer.save_model 只保存模型权重，不保存 optimizer
            #    我们需要用 _save_checkpoint 或直接 save_state 保存完整状态
            trainer.save_state()  # 保存到 output_dir/checkpoint-{global_step}/
            
            # 找到刚保存的最新 checkpoint 目录，作为下一个 chunk 的 resume 路径
            ckpt_dirs = [
                d for d in os.listdir(chunk_ckpt_dir) 
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(chunk_ckpt_dir, d))
            ]
            if ckpt_dirs:
                # 按 step 数排序，取最新的
                ckpt_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                resume_from = os.path.join(chunk_ckpt_dir, ckpt_dirs[-1])
                logging.info(f"💾 Checkpoint saved: {resume_from}")
                
                # 清理旧 checkpoint（只保留最新的一个，节省磁盘空间）
                for old_ckpt in ckpt_dirs[:-1]:
                    old_path = os.path.join(chunk_ckpt_dir, old_ckpt)
                    import shutil
                    shutil.rmtree(old_path, ignore_errors=True)
                    logging.info(f"🗑️ Removed old checkpoint: {old_path}")
            
            # 7. 打印缓存统计
            feature_cache.log_stats()
            
            logging.info(f"✅ Epoch {epoch+1}, Chunk {chunk_idx+1}/{num_chunks} done "
                         f"(global_step={trainer.state.global_step})")
    
    # 训练结束，保存最终模型权重（不含 optimizer，用于推理）
    trainer.save_model(args.save_model_path)
    feature_cache.shutdown()
    logging.info(f"✅ All epochs done. Final model saved to {args.save_model_path}")

else:
    # Stage2: 不需要 VGGT 特征缓存，正常训练
    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch_fused",
        bf16=False,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=grad_checkpointing,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=["wandb"],
        logging_dir='./logs/',
        logging_strategy='steps',
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

wandb.finish()

