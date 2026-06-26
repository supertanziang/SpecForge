# coding=utf-8
"""eagle3 VLM 在线多模态训练 collator: 复用 dflash 同款 OnlineMMDataset(读已 tokenize 的
{input_ids, loss_mask, image_file}), 在 dataloader worker 里 Image.open + processor
现编码 pixel_values, 训练 forward 直接吃。

为啥不预缓存 pixel_values:
    pixel_values 单条平均 ~5MB, 67919 条 = 340GB 磁盘, 还要 5 小时离线跑;
    NFS 剩 1.5TB, 半个盘就没了。在线 collate 完全没这成本, 训练 step 时机器
    本来就在等 GPU forward, CPU 顺手算 pixel_values 几乎免费。

为啥不会撞 fork-deadlock:
    PyTorch DataLoader 的 worker 在第一次 next(iter(loader)) 时才 fork, 此时主进程
    虽然已经 init 了 CUDA tensor, 但 PyTorch 已知这个问题, dataloader workers 默认
    用 spawn (linux 默认 fork, 但 worker 不会调 cuda 操作); collator 内部只用 PIL
    + transformers processor 的 CPU 路径, 不 touch CUDA, 安全。
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler

from .online_mm_dataset import OnlineMMDataset


class Eagle3OnlineMMCollator:
    """读取 OnlineMMDataset 产出的 {input_ids, loss_mask, image_file}, 现场算 pixel_values。

    输出与 specforge.data.utils.VlmDataCollatorWithPadding 期望一致:
        {
          "input_ids":      LongTensor[1, S],
          "attention_mask": LongTensor[1, S],
          "loss_mask":      LongTensor[1, S],
          "pixel_values":   FloatTensor[N_patches, D],
          "image_grid_thw": LongTensor[1, 3],
        }
    bs=1 only (与 dflash 一致, 多模态 batch 内拼接成本不划算)。
    """

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.image_processor = processor.image_processor
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1, "Eagle3 VLM online training only supports batch_size=1"
        f = features[0]
        input_ids = f["input_ids"][: self.max_length]
        loss_mask = f["loss_mask"][: self.max_length]
        image_file = f["image_file"]

        # 读图 + image processor 算 pixel_values / image_grid_thw
        image = Image.open(image_file).convert("RGB")
        image_inputs = self.image_processor(images=[image], return_tensors="pt")
        pixel_values = image_inputs.pixel_values  # [N_patches, D]
        image_grid_thw = image_inputs.image_grid_thw  # [1, 3]

        return {
            "input_ids": input_ids.unsqueeze(0).long(),  # [1, S]
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
            "loss_mask": loss_mask.unsqueeze(0).long(),  # [1, S]
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw.long(),
        }


def prepare_eagle3_mm_dataloader(
    dataset: OnlineMMDataset,
    processor,
    max_length: int = 4096,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    shuffle: bool = True,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """eagle3 VLM 在线训练 DataLoader (bs=1)。"""
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    if num_workers == 0:
        prefetch_factor = None
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=Eagle3OnlineMMCollator(processor, max_length=max_length),
        drop_last=True,
    )
