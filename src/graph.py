import json
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .agents import MASState, ModelRouter, creation_director_phase, critic_phase, generator_phase
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
        "director_creation",
        lambda state: creation_director_phase(state, config=config, router=router),
    )
    graph.add_node("generator_phase", lambda state: generator_phase(state, config=config, router=router))
    graph.add_node("sdxl_execution", _sdxl_node(sdxl))
    graph.add_node(
        "critic_phase",
        lambda state: critic_phase(state, config=config, router=router),
    )

    graph.add_edge(START, "director_creation")
    graph.add_edge("director_creation", "generator_phase")
    graph.add_edge("generator_phase", "sdxl_execution")
    graph.add_edge("sdxl_execution", "critic_phase")

    def decide_next(state: MASState) -> str:
        squad_status = state.get("squad_status") or {}
        all_done = all(v == "done" for v in squad_status.values()) if squad_status else False
        if all_done:
            return "end"
        if state.get("iteration", 0) >= config.max_loops:
            return "end"
        return "loop"

    graph.add_conditional_edges("critic_phase", decide_next, {"end": END, "loop": "generator_phase"})

    return graph.compile()
