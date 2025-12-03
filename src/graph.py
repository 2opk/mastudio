import json
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .agents import MASState, ModelRouter, director_phase, generator_phase
from .tools import SDXLWrapper
from .utils import SystemConfig


def _sdxl_node(sdxl: SDXLWrapper) -> Callable[[MASState], MASState]:
    def inner(state: MASState) -> MASState:
        prompts = state.get("generator_prompts", [])
        next_iteration = state.get("iteration", 0) + 1  # human-friendly round label starting at 1
        images = sdxl.generate(prompts, iteration=next_iteration)
        history = state.get("history", []) + [{"role": "sdxl", "content": json.dumps(images)}]
        return {**state, "images": images, "history": history, "iteration": next_iteration}

    return inner


def build_app(config: SystemConfig):
    router = ModelRouter(config)
    sdxl = SDXLWrapper(config.sdxl)

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

    graph.add_edge(START, "director_phase")
    graph.add_edge("generator_phase", "sdxl_execution")
    graph.add_edge("sdxl_execution", "director_phase")

    def decide_next(state: MASState) -> str:
        if state.get("iteration", 0) >= config.max_loops:
            return "end"
        return "loop"

    graph.add_conditional_edges("director_phase", decide_next, {"end": END, "loop": "generator_phase"})

    return graph.compile()
