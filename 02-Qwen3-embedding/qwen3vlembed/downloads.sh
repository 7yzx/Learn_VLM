mkdir -p ./datasets/conceptual_captions
modelscope download --dataset google-research-datasets/conceptual_captions --local_dir ./datasets/conceptual_captions

mkdir -p ./models/Qwen3-VL-Embedding-2B
modelscope download --model Qwen/Qwen3-VL-Embedding-2B --local_dir ./models/Qwen3-VL-Embedding-2B