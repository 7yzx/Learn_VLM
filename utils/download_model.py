import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

def down_load_model(repo_id, local_dir):
    """
    下载指定模型仓库到本地目录。

    参数:
    repo_id (str): 模型仓库的ID。
    local_dir (str): 本地保存模型的目录。
    """
    snapshot_download(
        repo_id=repo_id,  # 仓库ID
        local_dir=local_dir,             # 下载到本地的文件夹名称
        resume_download=True,              # 支持断点续传
        max_workers=8                      # 允许并发下载
    )
    
local_dir = '/mnt/sevenT/zixiaoy/checkpoints'

# repo_id = 'gongjy/minimind-v_dataset'
repo_id = "gongjy/minimind_dataset"

# repo_ids = ["LiheYoung/depth_anything_vitl14", "google/siglip2-large-patch16-384", "Qwen/Qwen2.5-VL-7B-Instruct"]
repo_ids = ["Qwen/Qwen2.5-VL-7B-Instruct","google-bert/bert-base-uncased" ]

for repo_id in repo_ids:
    
    datapath = os.path.join(local_dir, repo_id)
    down_load_model(repo_id=repo_id, local_dir=datapath)
