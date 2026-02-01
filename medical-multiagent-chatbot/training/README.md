Training (Adapters)

This folder contains a minimal QLoRA training pipeline using PEFT + Transformers.

Data format (JSONL):
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Example run:
python training/train_lora.py ^
  --base_model <model_id> ^
  --train_file <path_to_jsonl> ^
  --output_dir <adapter_output_dir> ^
  --max_seq_len 2048 ^
  --per_device_batch_size 1 ^
  --gradient_accumulation 8
