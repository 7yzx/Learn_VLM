import json
import os
import random
from bisect import bisect_right
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import datasets
from PIL import Image
from torch.utils.data import Dataset, Subset


DEFAULT_CAULDRON_DATASETS = [
    "chartqa",
    "finqa",
    "aokvqa",
    "figureqa",
    "diagram_image_to_text",
    "geomverse",
    "ai2d",
    "iam",
    "infographic_vqa",
    "intergps",
    "hateful_memes",
    "clevr",
    "iconqa",
    "multihiertt",
    "mapqa",
    "datikz",
    "hitab",
    "chart2text",
    "cocoqa",
    "docvqa",
    "dvqa",
]


class MixedSchemaMultiModalDataset(Dataset):
    def __init__(self, datasets_with_source: Sequence[Tuple[Dataset, str]]):
        self.datasets_with_source = list(datasets_with_source)
        self.cumulative_sizes: List[int] = []
        total_size = 0
        for dataset, _ in self.datasets_with_source:
            total_size += len(dataset)
            self.cumulative_sizes.append(total_size)

    def __len__(self) -> int:
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        local_index = index - previous_size
        dataset, source = self.datasets_with_source[dataset_idx]
        sample = dataset[local_index]
        if source == "minimind_v":
            return standardize_minimind_v_sample(sample)
        return standardize_cauldron_sample(sample)


def _load_cauldron_dataset(data_name: str, data_root: str):
    local_dataset_dir = os.path.join(data_root, data_name)
    if os.path.isdir(local_dataset_dir):
        parquet_pattern = os.path.join(local_dataset_dir, "*.parquet")
        return datasets.load_dataset("parquet", data_files=parquet_pattern)["train"]
    return datasets.load_dataset(data_root, data_name)["train"]


def load_cauldron_datasets(
    select_data: str,
    data_root: str = "data/the_cauldron",
    train_limit: Optional[int] = 60 * 1024,
):
    if select_data == "all":
        dataset_names = DEFAULT_CAULDRON_DATASETS
    elif select_data in DEFAULT_CAULDRON_DATASETS:
        dataset_names = [select_data]
    elif select_data == "none":
        dataset_names = []
    else:
        raise ValueError(f"cannot find dataset selection: {select_data}")

    dataset_list = []
    for data_name in dataset_names:
        try:
            dataset = _load_cauldron_dataset(data_name=data_name, data_root=data_root)
            dataset_list.append(dataset)
            print(f"成功加载 The Cauldron 子集: {data_name}, 样本数={len(dataset)}")
        except Exception as exc:
            print(f"加载 The Cauldron 子集失败: {data_name}, 错误: {exc}")

    if not dataset_list:
        return []

    if select_data == "all" and train_limit is not None:
        limited_list = []
        remaining = train_limit
        for dataset in dataset_list:
            if remaining <= 0:
                break
            take_count = min(len(dataset), remaining)
            limited_list.append(dataset.select(range(take_count)))
            remaining -= take_count
        dataset_list = limited_list

    return dataset_list


def load_minimind_v_dataset(parquet_path: str, max_samples: Optional[int] = None):
    if not parquet_path:
        raise ValueError("MiniMind-V 数据路径为空")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"未找到 MiniMind-V 数据文件: {parquet_path}，请先下载 minimind-v 的 parquet 数据。"
        )

    dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"成功加载 MiniMind-V 中文数据: {parquet_path}, 样本数={len(dataset)}")
    return dataset


def standardize_cauldron_sample(sample):
    return {
        "images": sample["images"][:1],
        "texts": sample["texts"],
    }


def _clean_user_text(text: str) -> str:
    cleaned = str(text).replace("<image>", " ").replace("\n\n", "\n").strip()
    return cleaned or "请根据图片回答问题。"


def _extract_prompt_pair(conversations) -> Tuple[str, str]:
    if isinstance(conversations, str):
        conversations = json.loads(conversations)

    user_text = None
    assistant_text = None
    for idx, turn in enumerate(conversations):
        role = turn.get("role")
        if role is None:
            role = "user" if idx % 2 == 0 else "assistant"
        content = turn.get("content", "")
        if role == "user" and user_text is None:
            user_text = _clean_user_text(content)
        elif role == "assistant" and assistant_text is None:
            assistant_text = str(content).strip()
        if user_text is not None and assistant_text is not None:
            break

    if user_text is None or assistant_text is None:
        raise ValueError("无法从 MiniMind-V 样本中解析 user/assistant 对话")

    return user_text, assistant_text


def standardize_minimind_v_sample(sample):
    user_text, assistant_text = _extract_prompt_pair(sample["conversations"])
    image = Image.open(BytesIO(sample["image_bytes"])).convert("RGB")
    return {
        "images": [image],
        "texts": [{"user": user_text, "assistant": assistant_text}],
    }


def build_mixed_training_dataset(
    select_data: str,
    data_seed: int,
    eval_sample_size: int = 64,
    cauldron_data_root: str = "data/the_cauldron",
    cauldron_train_limit: Optional[int] = 60 * 1024,
    include_chinese_data: bool = False,
    chinese_data_path: Optional[str] = None,
    chinese_data_limit: Optional[int] = None,
):
    dataset_sources: List[Tuple[Dataset, str]] = []

    for cauldron_dataset in load_cauldron_datasets(
        select_data=select_data,
        data_root=cauldron_data_root,
        train_limit=cauldron_train_limit,
    ):
        dataset_sources.append((cauldron_dataset, "cauldron"))

    if include_chinese_data:
        if not chinese_data_path:
            raise ValueError("启用中文数据混训时，必须提供 `chinese_data_path`")
        chinese_dataset = load_minimind_v_dataset(
            parquet_path=chinese_data_path,
            max_samples=chinese_data_limit,
        )
        dataset_sources.append((chinese_dataset, "minimind_v"))

    if not dataset_sources:
        raise ValueError("没有可用的训练数据源，请检查 The Cauldron 或 MiniMind-V 数据配置")

    merged_dataset = MixedSchemaMultiModalDataset(dataset_sources)
    all_indices = list(range(len(merged_dataset)))
    random.Random(data_seed).shuffle(all_indices)

    eval_size = min(eval_sample_size, max(1, len(all_indices) // 50), len(all_indices))
    if len(all_indices) <= 1:
        eval_size = 1
    elif eval_size >= len(all_indices):
        eval_size = max(1, len(all_indices) - 1)

    eval_indices = all_indices[:eval_size]
    train_indices = all_indices[eval_size:]

    if not train_indices:
        train_indices = eval_indices

    return {
        "train": Subset(merged_dataset, train_indices),
        "test": Subset(merged_dataset, eval_indices),
        "meta": {
            "total_samples": len(merged_dataset),
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "include_chinese_data": include_chinese_data,
            "chinese_data_path": chinese_data_path,
            "cauldron_selection": select_data,
        },
    }
