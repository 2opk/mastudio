import json
import os
from typing import Any, Dict, List, Optional, TypedDict

import litellm

from .utils import AgentMeta, SystemConfig, load_agents, load_prompt


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

    def _preview(self, text: str, limit: int = 200) -> str:
        if text is None:
            return ""
        text = str(text).replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _api_base_for_model(self, model: str) -> Optional[str]:
        if model.startswith("gemini/"):
            return os.getenv("GEMINI_API_BASE") or None
        if model == self.config.open_source_model:
            return self.config.vllm.api_base
        return None

    def call(self, *, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        resolved_model = self.config.resolve_model(model)
        kwargs = {
            "model": resolved_model,
            "api_base": self._api_base_for_model(resolved_model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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

        print(f"[LLM ->] model={resolved_model} api_base={kwargs['api_base'] or 'default'} temp={temperature}")
        print(f"  system: {self._preview(system_prompt)}")
        print(f"  user  : {self._preview(user_prompt)}")
        response = litellm.completion(**kwargs)
        content = response.choices[0].message["content"]
        print(f"[LLM <-] model={resolved_model} reply: {self._preview(content)}")
        return content


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.strip()}


def director_phase(state: MASState, config: SystemConfig, router: ModelRouter, director_path: str) -> MASState:
    print(f"[phase:director] iteration={state.get('iteration',0)} base_prompt={router._preview(state.get('user_prompt',''))}")
    directors = load_agents(director_path)
    notes: List[Dict[str, str]] = []
    for meta in directors:
        persona = load_prompt(meta.prompt)
        system_prompt = (
            f"{persona}\n"
            "You are collaborating with other directors. Produce concise notes as JSON with keys: "
            "insights (list), must_haves (list), risks (list), style (short string)."
        )
        user_prompt = (
            f"Base brief: {state.get('user_prompt','')}\n"
            f"Prior feedback: {state.get('feedback','none')}\n"
            "Respond with JSON only."
        )
        content = router.call(
            model=config.director_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.5
        )
        notes.append({"director": meta.id, "content": content})

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
    )
    creative_direction = _safe_json_parse(aggregated)
    history = state.get("history", []) + [{"role": "director", "content": aggregated}]
    return {**state, "creative_direction": creative_direction, "history": history, "feedback": ""}


def generator_phase(state: MASState, config: SystemConfig, router: ModelRouter, generator_path: str) -> MASState:
    print(f"[phase:generator] iteration={state.get('iteration',0)}")
    generators = load_agents(generator_path)
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
            model=config.generator_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.8
        )
        prompts.append(prompt.strip())
    history = state.get("history", []) + [{"role": "generator", "content": "\n".join(prompts)}]
    return {**state, "generator_prompts": prompts, "history": history}


def evaluation_phase(
    state: MASState, config: SystemConfig, router: ModelRouter, evaluator_prompt_path: str
) -> MASState:
    print(f"[phase:evaluator] iteration={state.get('iteration',0)}")
    evaluator_prompt = load_prompt(evaluator_prompt_path)
    system_prompt = evaluator_prompt
    user_prompt = (
        f"Iteration: {state.get('iteration', 0)}\n"
        f"Creative Direction: {json.dumps(state.get('creative_direction',{}), ensure_ascii=False)}\n"
        f"Prompts: {json.dumps(state.get('generator_prompts',[]), ensure_ascii=False)}\n"
        f"Images: {state.get('images', [])}\n"
        "Score and provide actionable feedback."
    )
    content = router.call(
        model=config.evaluator_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2
    )
    parsed = _safe_json_parse(content)
    score = parsed.get("score", 0)
    feedback = parsed.get("actionable_feedback", parsed.get("raw", content))
    scores = state.get("scores", []) + [score]
    history = state.get("history", []) + [{"role": "evaluator", "content": content}]
    iteration = state.get("iteration", 0) + 1
    return {**state, "feedback": feedback, "scores": scores, "history": history, "iteration": iteration}
