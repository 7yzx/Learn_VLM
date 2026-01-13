from modelscope.hub.snapshot_download import snapshot_download as scope_donwload
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

def download_dataset(repo_id, local_dir, filenames=None):
    """
    下载指定数据集仓库到本地目录。

    参数:
    repo_id (str): 数据集仓库的ID。
    local_dir (str): 本地保存数据集的目录。
    """
    if filenames is None:
        scope_donwload(
            repo_id=repo_id,  # 仓库ID
            repo_type="dataset",
            local_dir=local_dir,             # 下载到本地的文件夹名称
            max_workers=8                      # 允许并发下载
        )
    else:
        scope_donwload(
            repo_id=repo_id,  # 仓库ID
            allow_patterns=filenames,
            repo_type="dataset",
            local_dir=local_dir,             # 下载到本地的文件夹名称
            max_workers=8                      # 允许并发下载
        )


def download_hf_dataset(repo_id, local_dir, filenames=None):
    """
    下载指定数据集仓库到本地目录。

    参数:
    repo_id (str): 数据集仓库的ID。
    local_dir (str): 本地保存数据集的目录。
    """
    if filenames is None:
        snapshot_download(
            repo_id=repo_id,  # 仓库ID
            repo_type="dataset",
            local_dir=local_dir,             # 下载到本地的文件夹名称
            max_workers=8                      # 允许并发下载
        )
    else:
        snapshot_download(
            repo_id=repo_id,  # 仓库ID
            allow_patterns=filenames,
            repo_type="dataset",
            local_dir=local_dir,             # 下载到本地的文件夹名称
            max_workers=8                      # 允许并发下载
        )

local_dir = '/mnt/sevenT/zixiaoy/dataset'

# repo_id = 'gongjy/minimind-v_dataset'
# repo_id = "gongjy/minimind_dataset"
repo_id = "Inevitablevalor/MindCube"



datapath = os.path.join(local_dir, repo_id)
# download_dataset(repo_id=repo_id, local_dir=datapath)

download_hf_dataset(repo_id, datapath)

