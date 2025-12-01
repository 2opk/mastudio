import json
import os
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .agents import MASState, ModelRouter, director_phase, evaluation_phase, generator_phase
from .tools import SDXLWrapper
from .utils import SystemConfig


def _sdxl_node(sdxl: SDXLWrapper) -> Callable[[MASState], MASState]:
    def inner(state: MASState) -> MASState:
        prompts = state.get("generator_prompts", [])
        images = sdxl.generate(prompts)
        history = state.get("history", []) + [{"role": "sdxl", "content": json.dumps(images)}]
        return {**state, "images": images, "history": history}

    return inner


def build_app(config: SystemConfig):
    router = ModelRouter(config)
    sdxl = SDXLWrapper(config.sdxl, stability_key=os.getenv("STABILITY_API_KEY"))

    graph = StateGraph(MASState)
    graph.add_node(
        "director_phase",
        lambda state: director_phase(state, config=config, router=router, director_path="config/directors.json"),
    )
    graph.add_node(
        "generator_phase",
        lambda state: generator_phase(state, config=config, router=router, generator_path="config/generators.json"),
    )
    graph.add_node("sdxl_execution", _sdxl_node(sdxl))
    graph.add_node(
        "evaluation",
        lambda state: evaluation_phase(
            state, config=config, router=router, evaluator_prompt_path="prompts/evaluator.md"
        ),
    )

    graph.add_edge(START, "director_phase")
    graph.add_edge("director_phase", "generator_phase")
    graph.add_edge("generator_phase", "sdxl_execution")
    graph.add_edge("sdxl_execution", "evaluation")

    def decide_next(state: MASState) -> str:
        latest_score = state.get("scores", [0])[-1] if state.get("scores") else 0
        if latest_score >= config.evaluation_threshold:
            return "end"
        if state.get("iteration", 0) >= config.max_loops:
            return "end"
        return "loop"

    graph.add_conditional_edges(
        "evaluation",
        decide_next,
        {
            "end": END,
            "loop": "generator_phase",
        },
    )

    return graph.compile()
