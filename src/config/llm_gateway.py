"""LLM Gateway — abstracts Azure OpenAI calls with cost tracking and retry logic."""

import time
from enum import Enum

import tiktoken
from openai import AzureOpenAI

from src.config.settings import settings


class ModelTier(str, Enum):
    """Model routing tiers — cost-aware selection."""
    COMPLEX = "complex"     # GPT-4o: reasoning, evaluation, generation
    SIMPLE = "simple"       # GPT-4o-mini: classification, extraction


# Cost per 1K tokens (approximate, update as needed)
COST_PER_1K = {
    ModelTier.COMPLEX: {"input": 0.005, "output": 0.015},
    ModelTier.SIMPLE: {"input": 0.00015, "output": 0.0006},
}


class LLMGateway:
    """Centralized LLM access with cost tracking and retry logic."""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.per_agent_cost: dict[str, float] = {}

    def _get_deployment(self, tier: ModelTier) -> str:
        if tier == ModelTier.COMPLEX:
            return settings.azure_openai_deployment_gpt4
        return settings.azure_openai_deployment_gpt4_mini

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chat(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.COMPLEX,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        agent_name: str = "unknown",
        max_retries: int = 3,
    ) -> str:
        """Send chat completion with retry logic and cost tracking."""
        deployment = self._get_deployment(tier)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Track usage
                usage = response.usage
                if usage:
                    self.total_input_tokens += usage.prompt_tokens
                    self.total_output_tokens += usage.completion_tokens
                    cost = (
                        (usage.prompt_tokens / 1000) * COST_PER_1K[tier]["input"]
                        + (usage.completion_tokens / 1000) * COST_PER_1K[tier]["output"]
                    )
                    self.total_cost += cost
                    self.per_agent_cost[agent_name] = (
                        self.per_agent_cost.get(agent_name, 0) + cost
                    )
                
                self.call_count += 1
                return response.choices[0].message.content or ""
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM call failed after {max_retries} attempts: {e}"
                    ) from e
        
        return ""  # Should never reach here

    def get_cost_report(self) -> dict:
        """Return cost tracking summary."""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "per_agent_cost": {
                k: round(v, 4) for k, v in self.per_agent_cost.items()
            },
        }