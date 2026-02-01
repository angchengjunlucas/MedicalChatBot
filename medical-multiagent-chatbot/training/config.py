from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    train_file: str
    output_dir: str
    max_seq_len: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
