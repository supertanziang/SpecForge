# coding=utf-8
"""DFlash VLM 在线训练数据集。

读取 `scripts/convert_vispec_to_dflash.py` 产出的 JSONL(每行一个样本):
    {
      "input_ids":  list[int],   # 完整 token 序列(prompt + 目标生成), 已含图像占位 token
      "loss_mask":  list[int],   # 0=prompt(含图像 token), 1=生成段
      "image_file": str,         # 图片绝对路径
    }

设计参考 ViSpec `vispec/train/online_dataset.py` 的 OnlineMMDataset:
  - 数据集只返回「原始输入」(input_ids / loss_mask / image_file);
  - 目标模型 forward(取图像视觉特征 + 多层 hidden)在训练循环里实时做,
    见 `HFDFlashTargetModel.generate_dflash_data` 的 VLM 分支。

bs=1: 多模态每张图的 pixel_values 形状随图而异,batch 内拼接复杂,
与 ViSpec 一致只支持 batch_size=1。
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class OnlineMMDataset(Dataset):
    """读 JSONL 行,返回 input_ids[S] / loss_mask[S] / image_file。

    目标模型实时 forward 在训练循环里完成。
    """

    def __init__(self, jsonl_path: str, max_len: int = 4096):
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        input_ids = torch.tensor(rec["input_ids"][: self.max_len], dtype=torch.long)
        loss_mask = torch.tensor(rec["loss_mask"][: self.max_len], dtype=torch.long)
        return {
            "input_ids": input_ids,  # [S]
            "loss_mask": loss_mask,  # [S]
            "image_file": rec["image_file"],  # str
        }


class OnlineMMCollator:
    """bs=1 collator:input_ids/loss_mask 升为 [1, S],attention_mask 现生成全 1,image_file 透传。"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1, "DFlash VLM 训练仅支持 batch_size=1"
        f = features[0]
        input_ids = f["input_ids"].unsqueeze(0)  # [1, S]
        loss_mask = f["loss_mask"].unsqueeze(0)  # [1, S]
        attention_mask = torch.ones_like(input_ids)  # [1, S]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "image_file": f["image_file"],
        }


def prepare_mm_dataloader(
    dataset: OnlineMMDataset,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    shuffle: bool = True,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """为 VLM 在线训练构建 DataLoader(batch_size 固定为 1)。"""
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
        collate_fn=OnlineMMCollator(),
        drop_last=True,
    )
