import argparse
from pathlib import Path

from .graph import build_app
from .tools import save_json_report
from .utils import ensure_output_dir, load_env, load_system_config


def run(base_prompt: str):
    load_env()
    config = load_system_config()
    app = build_app(config)
    initial_state = {"user_prompt": base_prompt, "iteration": 0, "history": []}
    result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

    output_dir = ensure_output_dir(config.sdxl.output_dir)
    report_path = output_dir / "mas_report.json"
    save_json_report(
        {
            "base_prompt": base_prompt,
            "final_state": result,
        },
        report_path,
    )
    return result, report_path


def main():
    parser = argparse.ArgumentParser(description="Run Director–Generator MAS for generative art.")
    parser.add_argument("prompt", help="Base text prompt describing music/mood.")
    args = parser.parse_args()
    result, report_path = run(args.prompt)
    print(f"Run complete. Final iteration: {result.get('iteration')}. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
