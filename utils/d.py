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


def download_dataset(repo_id, local_dir, filenames=None):
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

def download_modelscope_dataset_files(dataset_id, local_dir, file_list):
    """
    下载 ModelScope 数据集中的多个特定文件
    """
    snapshot_download(
        dataset_id,
        repo_type='dataset',      # 明确指定是数据集 (dataset)
        local_dir=local_dir,      # 本地保存路径
        allow_patterns=file_list,  # 指定下载文件列表
        # ignore_file_pattern=[], # 也可以根据需要排除某些文件
    )
    print(f"下载完成！文件已保存至: {local_dir}")
    
    
def download_modelscope(dataset_id, local_dir, file):
    """
    下载 ModelScope 数据集中的多个特定文件
    """
    snapshot_download(
        dataset_id,
        repo_type='dataset',      # 明确指定是数据集 (dataset)
        local_dir=local_dir,      # 本地保存路径
        # ignore_file_pattern=[], # 也可以根据需要排除某些文件
    )
    print(f"下载完成！文件已保存至: {local_dir}")

if __name__ == "__main__":
    # data_repo_id = "jingyaogong/minimind_dataset"
    # filenames = ["pretrain_hq.jsonl", "dpo.jsonl", "rlaif-mini.jsonl", "sft_mini_512.jsonl"]
    # target_dir = "./minimind_dataset"
    data_repo_id = "AIGeeksGroup/Scene-30K"
    local_dir = '/mnt/sevenT/zixiaoy/dataset'

    datapath = os.path.join(local_dir, data_repo_id)
    # download_modelscope_dataset_files(data_repo_id, target_dir, filenames)
    
    download_dataset(data_repo_id, datapath)