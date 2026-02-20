"""
VGGT Feature Cache Manager — 分 Chunk 预加载版
================================================
核心思路：
  由于内存不够一次性放下全部 450GB 的 VGGT 特征，我们采用"分 chunk"策略：
  
  1. 把整个 dataset 按顺序切成 N 个 chunk（例如 2 个，前半和后半）
  2. 训练前，先把第 1 个 chunk 对应的 VGGT 特征全部预加载到内存
  3. 在这个 chunk 上跑若干个 epoch（chunk 内部正常 shuffle）
  4. 训练完后释放内存，加载第 2 个 chunk 的特征，继续训练
  5. 所有 chunk 轮完算一个"大 epoch"，可以重复多轮

  这样保证：
  - 训练时 100% 从内存读特征，零 NAS I/O，速度最快
  - 内存占用可控：只占 总量/N 的内存
  - chunk 内部有 shuffle，保留随机性

  唯一的 trade-off：
  - 同一个 chunk 内的样本会先被连续看到（局部性）
  - 但通过 chunk 内多 epoch shuffle + chunk 间交替，效果影响很小

使用方式：
  在 main.py 中通过 ChunkedTrainingManager 管理训练循环，
  替代原来的 trainer.train() 单次调用。
"""

import numpy as np
import torch
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import os
import gc
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VGGTFeatureCache:
    """
    VGGT 特征的内存缓存管理器。
    
    支持两种模式：
    1. bulk_load 模式：一次性预加载一批 idx 的特征到内存（推荐，配合分 chunk 使用）
    2. on-demand 模式：按需从 NAS 读取 + LRU 缓存（回退方案）

    Args:
        feature_dir: VGGT 特征的根目录 (NAS 路径)
        use_fp16: 是否在缓存中用 float16 存储（内存减半，读取时自动转 float32）
        num_workers: 预加载的并行线程数
    """

    def __init__(
        self,
        feature_dir: str,
        use_fp16: bool = True,
        num_workers: int = 4,
    ):
        self.feature_dir = feature_dir
        self.use_fp16 = use_fp16
        self.num_workers = num_workers

        # 内存缓存：idx -> np.ndarray
        self.cache: dict = {}
        self.cache_lock = threading.Lock()

        # 异步预取的线程池
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 统计
        self.hits = 0
        self.misses = 0

    def _load_from_disk(self, idx) -> np.ndarray:
        """从 NAS 读取单个特征文件"""
        path = os.path.join(self.feature_dir, str(idx), "vggt.npz")
        data = np.load(path)
        feature = data["feature"]  # [1, N=4, P_3D=1374, 2048]

        # 处理 NaN/Inf
        if np.isnan(feature).any() or np.isinf(feature).any():
            logger.warning(f"GT Data contains NaN/Inf at idx {idx}, replacing with 0")
            feature = np.nan_to_num(feature)

        if self.use_fp16:
            feature = feature.astype(np.float16)

        return feature

    def bulk_load(self, idx_list: list):
        """
        批量预加载一组 idx 的 VGGT 特征到内存。
        
        先清空旧缓存，释放内存，然后用多线程并行从 NAS 读取。
        这是分 chunk 训练的核心方法。

        Args:
            idx_list: 需要预加载的 idx 列表
        """
        # 1. 清空旧缓存，释放内存
        self.clear_cache()

        # 2. 多线程并行加载
        logger.info(f"🔄 Bulk loading {len(idx_list)} features from NAS (workers={self.num_workers})...")
        
        futures = {}
        for idx in idx_list:
            futures[idx] = self.executor.submit(self._load_from_disk, idx)

        # 等待所有加载完成，带进度条
        loaded = 0
        for idx in tqdm(idx_list, desc="Loading VGGT features", unit="samples"):
            try:
                feature = futures[idx].result()
                with self.cache_lock:
                    self.cache[idx] = feature
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load feature for idx={idx}: {e}")

        # 估算内存占用
        if self.use_fp16:
            mem_per_sample_mb = 22.5
        else:
            mem_per_sample_mb = 45.0
        total_mem_gb = loaded * mem_per_sample_mb / 1024
        logger.info(f"✅ Bulk loaded {loaded}/{len(idx_list)} features, ~{total_mem_gb:.1f}GB in memory")

    def clear_cache(self):
        """清空缓存并强制 GC 释放内存"""
        with self.cache_lock:
            count = len(self.cache)
            self.cache.clear()
        gc.collect()
        if count > 0:
            logger.info(f"🗑️ Cleared {count} cached features, memory released")

    def get(self, idx, device=None) -> torch.Tensor:
        """
        获取特征。优先从内存缓存读取，未命中则同步从 NAS 读取。

        Args:
            idx: 样本 index
            device: 目标 device (e.g., 'cuda:0')

        Returns:
            torch.Tensor, shape=[1,4,1374,2048], dtype=float32
        """
        feature = None

        # 1. 查缓存
        with self.cache_lock:
            if idx in self.cache:
                feature = self.cache[idx]
                self.hits += 1

        # 2. 缓存未命中，同步读取（回退方案，正常分 chunk 训练不应该走到这里）
        if feature is None:
            logger.warning(f"Cache miss for idx={idx}, falling back to NAS read (this should not happen in chunk mode)")
            feature = self._load_from_disk(idx)
            with self.cache_lock:
                self.cache[idx] = feature
            self.misses += 1

        # 转为 float32 tensor
        if self.use_fp16:
            tensor = torch.from_numpy(feature.astype(np.float32))
        else:
            tensor = torch.from_numpy(feature).float()

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def get_stats(self) -> dict:
        """返回缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

    def log_stats(self):
        """打印缓存统计"""
        stats = self.get_stats()
        logger.info(
            f"Feature Cache Stats: size={stats['cache_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hit_rate']}"
        )

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=False)


def split_dataset_into_chunks(dataset, num_chunks: int) -> list:
    """
    把 dataset 均匀切分成 num_chunks 个 chunk。
    
    Args:
        dataset: list 形式的 dataset（每个元素是一个 conversation sample）
        num_chunks: 切分数量
    
    Returns:
        list of lists，每个子 list 是一个 chunk 的样本
    """
    total = len(dataset)
    chunk_size = (total + num_chunks - 1) // num_chunks  # 向上取整
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start < total:
            chunks.append(dataset[start:end])
    logger.info(f"Dataset split into {len(chunks)} chunks: {[len(c) for c in chunks]}")
    return chunks


def extract_idx_from_chunk(chunk) -> list:
    """
    从一个 chunk 的样本中提取所有 idx。
    
    你的 dataset 中每个样本是一个 conversation list，第一个元素是 {"idx": ...}。
    """
    idx_list = []
    for sample in chunk:
        if isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict) and "idx" in sample[0]:
            idx_list.append(sample[0]["idx"])
        elif isinstance(sample, dict) and "idx" in sample:
            idx_list.append(sample["idx"])
    return idx_list
