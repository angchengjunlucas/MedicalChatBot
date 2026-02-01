from __future__ import annotations

from dataclasses import dataclass

from .base import LLMProvider, LLMResponse, LLMMessage


@dataclass(frozen=True)
class ModelRouter:
    provider: LLMProvider
    default_model: str
    supervisor_model: str
    cardiology_model: str
    geriatrics_model: str
    mental_model: str

    async def generate(
        self,
        role: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if hasattr(self.provider, "generate_with_role"):
            return await self.provider.generate_with_role(
                role,
                messages,
                temperature,
                max_tokens,
            )
        model_id = self._resolve_model(role)
        return await self.provider.generate(model_id, messages, temperature, max_tokens)

    def _resolve_model(self, role: str) -> str:
        mapping = {
            "supervisor": self.supervisor_model,
            "cardiology": self.cardiology_model,
            "geriatrics": self.geriatrics_model,
            "mental": self.mental_model,
        }
        return mapping.get(role, self.default_model)

# this is the traffic controller for my models, this does not determine the order of the models called.
