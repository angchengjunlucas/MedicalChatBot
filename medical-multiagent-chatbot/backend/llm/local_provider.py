from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import LLMMessage, LLMProvider, LLMResponse, LLMUsage


@dataclass
class LocalPeftProvider(LLMProvider):
    base_model_id: str
    adapter_paths: dict[str, str]
    device_map: str = "auto"

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        quant_config = _build_quant_config(dtype)
        self._offload_folder = _resolve_offload_folder(self.device_map)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.device_map,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        if self._offload_folder:
            load_kwargs["offload_folder"] = self._offload_folder

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            **load_kwargs,
        )
        self._model = self._load_adapters(self._base_model, self.adapter_paths)
        self._model.eval()

    async def generate(
        self,
        model_id: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        role = model_id
        return await self.generate_with_role(role, messages, temperature, max_tokens)

    async def generate_with_role(
        self,
        role: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync,
                role,
                messages,
                temperature,
                max_tokens,
            )

    def _generate_sync(
        self,
        role: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        adapter_name = self._adapter_for_role(role)
        model_ctx = _adapter_context(self._model, adapter_name)
        with model_ctx:
            prompt_inputs = _prepare_inputs(self._tokenizer, messages)
            input_ids = prompt_inputs["input_ids"]
            attention_mask = prompt_inputs.get("attention_mask")

            do_sample = True
            if temperature <= 0:
                temperature = 0.7
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "eos_token_id": self._tokenizer.eos_token_id,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                outputs = self._model.generate(input_ids=input_ids, **gen_kwargs)

            generated = outputs[0][input_ids.shape[-1]:]
            content = self._tokenizer.decode(generated, skip_special_tokens=True)

            prompt_tokens = int(input_ids.shape[-1])
            completion_tokens = int(generated.shape[-1])
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            return LLMResponse(
                content=content.strip(),
                model=self.base_model_id,
                usage=usage,
                raw=None,
            )

    def _adapter_for_role(self, role: str) -> str | None:
        if role in self._role_to_adapter:
            return self._role_to_adapter[role]
        return None

    def _load_adapters(
        self,
        base_model: Any,
        adapter_paths: dict[str, str],
    ) -> PeftModel | Any:
        if not adapter_paths:
            self._role_to_adapter = {}
            return base_model

        adapter_items = list(adapter_paths.items())
        first_role, first_path = adapter_items[0]

        kwargs: dict[str, Any] = {"is_trainable": False}
        if "adapter_name" in inspect.signature(PeftModel.from_pretrained).parameters:
            kwargs["adapter_name"] = first_role
            default_adapter_name = first_role
        else:
            default_adapter_name = "default"
        if self._offload_folder and "offload_folder" in inspect.signature(PeftModel.from_pretrained).parameters:
            kwargs["offload_folder"] = self._offload_folder

        model = PeftModel.from_pretrained(base_model, first_path, **kwargs)
        role_to_adapter = {first_role: default_adapter_name}

        for role, path in adapter_items[1:]:
            load_adapter = getattr(model, "load_adapter", None)
            if not callable(load_adapter):
                continue
            load_kwargs: dict[str, Any] = {"adapter_name": role} if "adapter_name" in inspect.signature(load_adapter).parameters else {}
            if self._offload_folder and "offload_folder" in inspect.signature(load_adapter).parameters:
                load_kwargs["offload_folder"] = self._offload_folder
            model.load_adapter(path, **load_kwargs)
            role_to_adapter[role] = role

        self._role_to_adapter = role_to_adapter
        return model


def _prepare_inputs(tokenizer: Any, messages: list[LLMMessage]) -> dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            return {"input_ids": encoded.to(_device())}
        return {k: v.to(_device()) for k, v in encoded.items()}

    prompt = _messages_to_prompt(messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(_device()) for k, v in encoded.items()}


def _messages_to_prompt(messages: list[LLMMessage]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_quant_config(dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if not torch.cuda.is_available():
        return None
    try:
        from bitsandbytes.nn import Linear4bit
        _ = Linear4bit(4, 4, bias=False).to("cuda")
    except Exception:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def _resolve_offload_folder(device_map: str) -> str | None:
    if device_map != "auto":
        return None
    folder = os.getenv("LOCAL_OFFLOAD_DIR", "./data/offload")
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class _adapter_context:
    def __init__(self, model: Any, adapter_name: str | None) -> None:
        self._model = model
        self._adapter_name = adapter_name
        self._ctx = None

    def __enter__(self):
        if self._adapter_name and hasattr(self._model, "set_adapter"):
            self._model.set_adapter(self._adapter_name)
            return self._model
        if hasattr(self._model, "disable_adapter"):
            self._ctx = self._model.disable_adapter()
            return self._ctx.__enter__()
        return self._model

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False
