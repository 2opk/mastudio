import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AgentMeta(BaseModel):
    id: str
    name: str
    prompt: str


class VLLMConfig(BaseModel):
    api_base: str = Field(default_factory=lambda: os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))
    tensor_parallel_size: int = 4


class SDXLConfig(BaseModel):
    mode: str = "mock"  # mock | stability
    output_dir: str = "output"


class SystemConfig(BaseModel):
    evaluation_threshold: int
    max_loops: int
    recursion_limit: int = 1000
    models: Dict[str, str] = Field(default_factory=dict)
    aliases: Dict[str, str] = Field(default_factory=dict)
    vllm: VLLMConfig
    sdxl: SDXLConfig

    def resolve_model(self, key_or_model: str) -> str:
        """Return a LiteLLM-ready model string for a given alias or direct name."""
        if not key_or_model:
            return ""
        return self.aliases.get(key_or_model, key_or_model)

    @property
    def director_model(self) -> str:
        return self.resolve_model(self.models.get("director", ""))

    @property
    def generator_model(self) -> str:
        return self.resolve_model(self.models.get("generator", ""))

    @property
    def evaluator_model(self) -> str:
        return self.resolve_model(self.models.get("evaluator", ""))

    @property
    def open_source_model(self) -> str:
        return self.resolve_model(self.models.get("open_source", ""))


def load_env() -> None:
    """Load .env variables once."""
    load_dotenv()


def load_agents(path: str) -> List[AgentMeta]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [AgentMeta(**item) for item in data]


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_config(path: str = "config/system_config.yaml") -> SystemConfig:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict = yaml.safe_load(f)
    # Allow overriding vLLM endpoint via env even if YAML ships with a placeholder.
    if isinstance(data.get("vllm"), dict):
        env_vllm_base: Optional[str] = os.getenv("VLLM_API_BASE")
        if env_vllm_base:
            data["vllm"]["api_base"] = env_vllm_base
    return SystemConfig(**data)


def ensure_output_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
