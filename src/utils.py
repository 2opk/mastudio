import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AgentMeta(BaseModel):
    id: str
    name: str
    prompt: str
    display_name: Optional[str] = None


class VLLMConfig(BaseModel):
    api_base: str = Field(default_factory=lambda: os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))
    tensor_parallel_size: int = 4


class SDXLConfig(BaseModel):
    mode: str = "mock"  # mock | api | local
    output_dir: str = "output"
    device: str = "cuda:0"


class SystemConfig(BaseModel):
    evaluation_threshold: int
    max_loops: int
    recursion_limit: int = 2500
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


def load_agent_aliases(path: str = "prompts/agents.yaml") -> Dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    agents = raw.get("agents", [])
    aliases: Dict[str, str] = {}
    for item in agents:
        agent_id = item.get("id")
        display = item.get("display_name") or item.get("name")
        if agent_id and display:
            aliases[agent_id] = display
    return aliases


def load_agents(path: str, aliases: Optional[Dict[str, str]] = None) -> List[AgentMeta]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alias_map = aliases or {}
    processed: List[AgentMeta] = []
    for item in data:
        agent_id = item.get("id", "")
        display_name = alias_map.get(agent_id) or item.get("display_name") or item.get("name")
        processed.append(AgentMeta(**item, display_name=display_name))
    return processed


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


def prepare_run_dirs(base_output_dir: str) -> Tuple[Path, Path, str]:
    """Create a timestamped run dir and intms subdir under the base output dir, without clearing prior runs."""
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = base / f"output_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    intms_dir = run_dir / "intms"
    intms_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, intms_dir, timestamp
