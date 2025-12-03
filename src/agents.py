import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import litellm
from rich.console import Console
from rich.text import Text

from .utils import AgentMeta, SystemConfig, load_agent_aliases, load_agents, load_prompt


class MASState(TypedDict, total=False):
    iteration: int
    user_prompt: str
    creative_direction: Dict[str, Any]
    generator_prompts: List[str]
    images: List[str]
    feedback: str
    scores: List[float]
    history: List[Dict[str, str]]


class ModelRouter:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.console = Console()
        self._logged_models = set()

    def _preview(self, text: Any, limit: int = 1000) -> str:
        if text is None:
            return ""
        if isinstance(text, list):
            texts: List[str] = []
            for part in text:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = str(part.get("text", ""))
                    texts.append(t)
            combined = "\n\n".join(texts) if texts else f"[multimodal {len(text)} parts]"
            return self._sanitize_preview(combined, limit)
        return self._sanitize_preview(str(text), limit)

    def _sanitize_preview(self, raw: str, limit: int) -> str:
        s = raw.replace("\\n", "\n").replace('\\"', '"')
        return s if len(s) <= limit else s[: limit - 3] + "..."

    def _api_base_for_model(self, model: str) -> Optional[str]:
        if model.startswith("gemini/"):
            return os.getenv("GEMINI_API_BASE") or None
        if model == self.config.open_source_model:
            return self.config.vllm.api_base
        return None

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        agent_label: Optional[str] = None,
        user_content: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        resolved_model = self.config.resolve_model(model)
        content = user_content if user_content is not None else user_prompt
        kwargs = {
            "model": resolved_model,
            "api_base": self._api_base_for_model(resolved_model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
        }
        # Gemini models can require an explicit API version; allow overriding via env.
        if resolved_model.startswith("gemini/"):
            kwargs["api_version"] = os.getenv("GEMINI_API_VERSION", "v1")
        # Local OpenAI-compatible endpoints (vLLM) usually need a dummy API key.
        if kwargs["api_base"] and not kwargs.get("api_key"):
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY") or "EMPTY"
        # If talking to an OpenAI-compatible endpoint, force provider for litellm.
        if kwargs["api_base"] and not resolved_model.startswith(("gemini/", "anthropic/", "openai/")):
            kwargs["custom_llm_provider"] = "openai"

        self._log_model_once(resolved_model, kwargs["api_base"])
        response = litellm.completion(**kwargs)
        content = response.choices[0].message["content"]
        if agent_label:
            self._print_chat_line(agent_label, "", self._preview(content))
        return content

    def _log_model_once(self, model: str, api_base: Optional[str]) -> None:
        key = (model, api_base)
        if key in self._logged_models:
            return
        self._logged_models.add(key)
        line = Text(f"[model] {model}", style="dim")
        if api_base:
            line.append(f" via {api_base}", style="dim")
        self.console.print(line)

    def _print_chat_line(self, agent_label: str, direction: str, text: str) -> None:
        header = Text(f"[{agent_label}] {direction}", style="bold cyan" if direction == "→" else "bold magenta")
        body = Text(f"{text}", style="grey70")
        self.console.print(header)
        self.console.print(body)
        self.console.print()  # blank line for readability


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.strip()}


def director_phase(state: MASState, config: SystemConfig, router: ModelRouter, director_path: str) -> MASState:
    router.console.print(
        Text(
            f"[phase:director] iteration={state.get('iteration',0)} base_prompt={router._preview(state.get('user_prompt',''))}",
            style="bold white",
        )
    )
    aliases = load_agent_aliases()
    directors = load_agents(director_path, aliases)
    notes: List[Dict[str, str]] = []
    history_entries: List[Dict[str, str]] = state.get("history", []).copy()
    images = state.get("images", [])
    text_body = (
        f"Base brief: {state.get('user_prompt','')}\n"
        f"Prior feedback: {state.get('feedback','none')}\n"
        f"Images included: {len(images)}. Review if present.\n"
        "Respond with JSON only."
    )
    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": text_body}]
    for img_path in images:
        p = Path(img_path)
        if not p.exists():
            continue
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        except OSError:
            continue
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    for meta in directors:
        persona = load_prompt(meta.prompt)
        system_prompt = (
            f"{persona}\n"
            "You are collaborating with other directors. Produce concise notes as JSON with keys: "
            "insights (list), must_haves (list), risks (list), style (short string)."
        )
        content = router.call(
            model=config.director_model,
            system_prompt=system_prompt,
            user_prompt=text_body,
            temperature=0.5,
            agent_label=meta.display_name or meta.name,
            user_content=content_parts,
        )
        notes.append({"director": meta.id, "content": content})
        history_entries.append({"role": meta.display_name or meta.name, "content": content})

    aggregate_input = "\n".join([f"{n['director']}: {n['content']}" for n in notes])
    aggregate_system = (
        "You merge director feedback into one Creative Direction JSON with keys: "
        "theme, mood, palette, composition, focal_points, must_haves, avoid, story_hook."
    )
    aggregated = router.call(
        model=config.director_model,
        system_prompt=aggregate_system,
        user_prompt=f"Combine these director notes:\n{aggregate_input}",
        temperature=0.3,
        agent_label=aliases.get("director_aggregate", "Creative Director"),
    )
    creative_direction = _safe_json_parse(aggregated)
    history = history_entries + [{"role": aliases.get("director_aggregate", "Creative Director"), "content": aggregated}]
    return {**state, "creative_direction": creative_direction, "history": history, "feedback": aggregated}


def generator_phase(state: MASState, config: SystemConfig, router: ModelRouter, generator_path: str) -> MASState:
    router.console.print(Text(f"[phase:generator] iteration={state.get('iteration',0)}", style="bold white"))
    aliases = load_agent_aliases()
    generators = load_agents(generator_path, aliases)
    history_entries: List[Dict[str, str]] = state.get("history", []).copy()
    prompts: List[str] = []
    for meta in generators:
        persona = load_prompt(meta.prompt)
        system_prompt = (
            f"{persona}\n"
            "Write one SDXL prompt that is direct, visual, and camera-ready. "
            "Include subject, setting, lighting, color, lens, and style. 40 words max."
        )
        user_prompt = (
            f"Base brief: {state.get('user_prompt','')}\n"
            f"Creative Direction: {json.dumps(state.get('creative_direction',{}), ensure_ascii=False)}\n"
            f"Feedback: {state.get('feedback','none')}\n"
            "Return only the prompt text."
        )
        prompt = router.call(
            model=config.generator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,
            agent_label=meta.display_name or meta.name,
        )
        prompts.append(prompt.strip())
        history_entries.append({"role": meta.display_name or meta.name, "content": prompt.strip()})
    history = history_entries + [{"role": "Generator Aggregate", "content": "\n".join(prompts)}]
    return {**state, "generator_prompts": prompts, "history": history}


def evaluation_phase(
    state: MASState, config: SystemConfig, router: ModelRouter, evaluator_prompt_path: str
) -> MASState:
    router.console.print(Text(f"[phase:evaluator] iteration={state.get('iteration',0)}", style="bold white"))
    aliases = load_agent_aliases()
    evaluator_prompt = load_prompt(evaluator_prompt_path)
    system_prompt = evaluator_prompt
    text_part = (
        f"Iteration: {state.get('iteration', 0)}\n"
        f"Creative Direction: {json.dumps(state.get('creative_direction',{}), ensure_ascii=False)}\n"
        f"Prompts: {json.dumps(state.get('generator_prompts',[]), ensure_ascii=False)}\n"
        "Score and provide actionable feedback."
    )
    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": text_part}]
    for img_path in state.get("images", []):
        p = Path(img_path)
        if not p.exists():
            continue
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        except OSError:
            continue
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content = router.call(
        model=config.evaluator_model,
        system_prompt=system_prompt,
        user_prompt=text_part,
        temperature=0.2,
        agent_label=aliases.get("evaluator", "Evaluator"),
        user_content=content_parts,
    )
    parsed = _safe_json_parse(content)
    score = parsed.get("score", 0)
    feedback = parsed.get("actionable_feedback", parsed.get("raw", content))
    scores = state.get("scores", []) + [score]
    history = state.get("history", []) + [{"role": aliases.get("evaluator", "Evaluator"), "content": content}]
    iteration = state.get("iteration", 0) + 1
    return {**state, "feedback": feedback, "scores": scores, "history": history, "iteration": iteration}
