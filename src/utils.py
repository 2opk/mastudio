import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AgentMeta(BaseModel):
    id: str
    name: str
    prompt: str
    display_name: Optional[str] = None


class DirectorMeta(BaseModel):
    id: str
    name: str
    creation_prompt: str
    critic_prompt: str
    display_name: Optional[str] = None


class GeneratorAgentMeta(BaseModel):
    id: str
    name: str
    prompt: str
    display_name: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class VLLMConfig(BaseModel):
    api_base: str = Field(default_factory=lambda: os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))
    tensor_parallel_size: int = 4


class SDXLConfig(BaseModel):
    mode: str = "mock"  # mock | api | local
    output_dir: str = "output"
    device: str = "cuda:0"
    api_base: Optional[str] = Field(default_factory=lambda: os.getenv("SDXL_API_BASE", "") or "")
    failover_mode: str = "none"  # none | mock


class SystemConfig(BaseModel):
    evaluation_threshold: int
    max_loops: int
    creative_index_threshold: float = 4.0
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


def load_directors(path: str, aliases: Optional[Dict[str, str]] = None) -> List[DirectorMeta]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alias_map = aliases or {}
    processed: List[DirectorMeta] = []
    for item in data:
        if not all(k in item for k in ("id", "name", "creation_prompt", "critic_prompt")):
            continue
        agent_id = item.get("id", "")
        display_name = alias_map.get(agent_id) or item.get("display_name") or item.get("name")
        processed.append(DirectorMeta(**item, display_name=display_name))
    return processed


def load_directors_auto(aliases: Optional[Dict[str, str]] = None) -> List[DirectorMeta]:
    """Load director metas without relying on config files."""
    alias_map = aliases or {}
    base = Path("prompts/directors")
    mapping = [
        ("music_interpreter", "Music Interpreter"),
        ("visual_director", "Visual Director"),
        ("concept_architect", "Concept Architect"),
        ("orchestrator", "Orchestrator"),
    ]
    metas: List[DirectorMeta] = []
    for dir_id, name in mapping:
        creation_prompt = base / "creation_mode" / f"{dir_id}.md"
        critic_prompt = base / "critic_mode" / f"{dir_id}.md"
        metas.append(
            DirectorMeta(
                id=dir_id,
                name=name,
                creation_prompt=str(creation_prompt),
                critic_prompt=str(critic_prompt),
                display_name=alias_map.get(dir_id, name),
            )
        )
    return metas


def _extract_persona_name(raw: str) -> Optional[str]:
    match = re.search(r"##\s*PERSONA:\s*(.+)", raw)
    if match:
        return match.group(1).strip()
    return None


def _extract_keywords(raw: str) -> List[str]:
    """Extract backtick-wrapped keywords from persona prompt."""
    keywords: List[str] = []
    for match in re.findall(r"`([^`]+)`", raw):
        parts = [p.strip() for p in re.split(r"[;,]", match) if p.strip()]
        if parts:
            keywords.extend(parts)
        else:
            keywords.append(match.strip())
    return keywords


def load_generator_pool(root: str = "prompts/generators") -> List[GeneratorAgentMeta]:
    base = Path(root)
    if not base.exists():
        return []
    exclude_dirs = {"creation_mode", "refinement_mode"}
    metas: List[GeneratorAgentMeta] = []
    for path in base.rglob("*.md"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        rel_parts = path.relative_to(base).parts
        category = rel_parts[0] if rel_parts else None
        raw = path.read_text(encoding="utf-8")
        name = _extract_persona_name(raw) or path.stem.replace("_", " ").title()
        keywords = _extract_keywords(raw)
        metas.append(
            GeneratorAgentMeta(
                id=path.stem,
                name=name,
                prompt=str(path),
                display_name=name,
                keywords=keywords,
                category=category,
            )
        )
    return metas


def stable_seed(text: str) -> int:
    """Return deterministic seed from text for squad selection randomness."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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
    # Normalize SDXL api_base to a string.
    if isinstance(data.get("sdxl"), dict):
        api_base = data["sdxl"].get("api_base")
        data["sdxl"]["api_base"] = api_base or os.getenv("SDXL_API_BASE", "") or ""
        failover_env = os.getenv("SDXL_FAILOVER_MODE")
        if failover_env:
            data["sdxl"]["failover_mode"] = failover_env
    return SystemConfig(**data)


def ensure_output_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_run_dirs(base_output_dir: str, use_as_run_dir: bool = False) -> Tuple[Path, Path, str]:
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = base if use_as_run_dir else base / f"output_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    intms_dir = run_dir / "intms"
    intms_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, intms_dir, timestamp
