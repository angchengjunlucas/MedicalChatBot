from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from training.config import TrainConfig
from training.data_format import format_for_sft, load_chat_jsonl


def _can_use_4bit() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb
    except Exception:
        return False
    try:
        from bitsandbytes.nn import Linear4bit
        _ = Linear4bit(4, 4, bias=False).to("cuda")
        return True
    except Exception:
        return False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    examples = load_chat_jsonl(cfg.train_file)
    texts = [format_for_sft(ex.messages) for ex in examples]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            max_length=cfg.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = [{"text": t} for t in texts]
    tokenized = [tokenize(item) for item in dataset]

    use_4bit = _can_use_4bit()
    if not use_4bit:
        print("4-bit quantization disabled (no CUDA-capable bitsandbytes).")

    device_map = "auto"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        load_in_4bit=use_4bit,
        device_map=device_map,
        torch_dtype=dtype,
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
