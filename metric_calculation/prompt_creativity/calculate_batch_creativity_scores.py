"""
Calculate Creativity Scores for Batch-Generated Visual Prompts (Lite Version)
Compatible with flat-list JSON structure. Also supports aggregation of existing
score files by prompt_id (0/1/2 => harmonic/conflict/random).
"""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from creativity_evaluator import (
        MusicToImageCreativityEvaluator,
        CreativityMetrics,
    )
except ModuleNotFoundError:
    MusicToImageCreativityEvaluator = None  # type: ignore
    CreativityMetrics = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calculate_creativity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PromptCreativityScore:
    """Individual prompt creativity score"""
    sample_idx: int
    prompt_id: int
    mode: str
    temperature: float
    originality: float
    elaboration: float
    alignment: float
    coherence: float
    overall: float
    prompt_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BatchCreativityCalculator:
    """Calculate creativity scores for batch-generated prompts"""

    def __init__(
        self,
        prompts_file: str = "prompts_for_image_generation.json",
        output_dir: str = "creativity_scores",
        use_llm: bool = False,
        api_key: str = None
    ):
        self.prompts_file = Path(prompts_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if MusicToImageCreativityEvaluator is None:
            raise ImportError("creativity_evaluator is required for scoring but is not installed.")

        # Load prompts
        logger.info(f"Loading prompts from {prompts_file}")
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Validate structure (Expect List)
        if isinstance(self.data, list):
            self.items = self.data
            logger.info(f"Detected flat list structure. Loaded {len(self.items)} items.")
        elif isinstance(self.data, dict) and 'samples' in self.data:
             # Fallback for nested structure (if needed in future)
            self.items = []
            for sample in self.data['samples']:
                features = sample.get('musical_features', {})
                for p in sample.get('prompts', []):
                    p['features'] = features # Flatten for consistent processing
                    p['sample_idx'] = sample.get('sample_idx')
                    self.items.append(p)
            logger.info(f"Detected nested structure. Flattened to {len(self.items)} items.")
        else:
            raise ValueError("Unknown JSON structure. Expected a list of objects or {'samples': [...]}")

        # Initialize Evaluator
        self.evaluator = MusicToImageCreativityEvaluator(use_llm=use_llm, api_key=api_key)
        self.prompt_scores: List[PromptCreativityScore] = []

    async def calculate_prompt_scores(self) -> List[PromptCreativityScore]:
        """
        Calculate creativity scores for all prompts in the list.
        """
        mode_name = 'GPT-4 Judge' if self.evaluator.use_llm else 'Heuristic Rules'
        logger.info(f"Calculating scores (Mode: {mode_name})...")
        
        self.prompt_scores = []
        total = len(self.items)

        for idx, item in enumerate(self.items):
            if idx % 10 == 0:
                logger.info(f"Processing item {idx + 1}/{total}")

            # Extract Data
            visual_prompt = item.get('visual_prompt', '')
            # JSON Key is 'features', code expects 'musical_features' logic but passed as dict
            musical_features = item.get('features', {}) 
            
            # Calculate score
            metrics = await self.evaluator.evaluate(visual_prompt, musical_features)

            # Create score object
            score = PromptCreativityScore(
                sample_idx=item.get('sample_idx', -1),
                prompt_id=item.get('prompt_id', -1),
                mode=item.get('mode', 'unknown'),
                temperature=item.get('temperature', 0.0),
                originality=metrics.originality,
                elaboration=metrics.elaboration,
                alignment=metrics.alignment,
                coherence=metrics.coherence,
                overall=metrics.overall,
                prompt_text=visual_prompt
            )

            self.prompt_scores.append(score)

        logger.info(f"✓ Calculated {len(self.prompt_scores)} prompt scores")
        return self.prompt_scores

    def save_prompt_scores(self, filename: str = "creativity_prompt_scores.json") -> Path:
        """Save individual prompt scores."""
        filepath = self.output_dir / filename
        data = [s.to_dict() for s in self.prompt_scores]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved prompt scores to {filepath}")
        return filepath


PROMPT_ID_LABELS = {0: "harmonic", 1: "conflict", 2: "random"}
METRIC_KEYS = ["originality", "elaboration", "alignment", "coherence", "overall"]


def load_scores(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def aggregate_by_prompt_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Return averages grouped by prompt_id label (harmonic/conflict/random)."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        pid = item.get("prompt_id")
        label = PROMPT_ID_LABELS.get(pid)
        if label is None:
            continue
        grouped[label].append(item)

    result: Dict[str, Dict[str, float]] = {}
    for label, rows in grouped.items():
        result[label] = {}
        for key in METRIC_KEYS:
            vals = [float(r.get(key, 0.0)) for r in rows if r.get(key) is not None]
            result[label][key] = sum(vals) / len(vals) if vals else 0.0
    return result


def aggregate_multiple(label_paths: List[Tuple[str, Path]], output_path: Path) -> Path:
    """Aggregate multiple score files into one summary JSON."""
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, path in label_paths:
        scores = load_scores(path)
        summary[label] = aggregate_by_prompt_id(scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved aggregated averages to {output_path}")
    return output_path


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate creativity scores for batch prompts")
    parser.add_argument(
        "--prompts-file",
        default="prompts_for_image_generation.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="creativity_scores",
        help="Output directory for results"
    )
    parser.add_argument("--use-llm", action="store_true", help="Use GPT-4 as judge")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument(
        "--aggregate-scores",
        action="append",
        default=[],
        help="Aggregate existing score files as label:path (e.g., qwen:output_qwen/results/prompt_creativity/creativity_prompt_scores.json). Can be given multiple times.",
    )
    parser.add_argument(
        "--aggregate-output",
        default="results/prompt_creativity/creativity_prompt_group_avgs.json",
        help="Path to write aggregated averages JSON.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip scoring and only aggregate existing score files.",
    )

    args = parser.parse_args()

    # Optional: run scoring
    if not args.aggregate_only:
        try:
            calculator = BatchCreativityCalculator(
                prompts_file=args.prompts_file,
                output_dir=args.output_dir,
                use_llm=args.use_llm,
                api_key=args.api_key
            )
        except Exception as e:
            logger.error(f"Initialization Error: {e}")
            return 1

        try:
            await calculator.calculate_prompt_scores()
            calculator.save_prompt_scores()
            
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE")
            print(f"Total prompts scored: {len(calculator.prompt_scores)}")
            print(f"Results saved to: {args.output_dir}")
            print("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Aggregation step (defaults to qwen/chatgpt if available)
    label_paths: List[Tuple[str, Path]] = []
    if args.aggregate_scores:
        for entry in args.aggregate_scores:
            if ":" not in entry:
                logger.error(f"Invalid aggregate-scores entry (expected label:path): {entry}")
                return 1
            lbl, path_str = entry.split(":", 1)
            label_paths.append((lbl.strip(), Path(path_str.strip())))
    else:
        defaults = [
            ("qwen", Path("output_qwen/results/prompt_creativity/creativity_prompt_scores.json")),
            ("gpt", Path("output_chatgpt/results/prompt_creativity/creativity_prompt_scores.json")),
        ]
        for lbl, p in defaults:
            if p.exists():
                label_paths.append((lbl, p))

    if label_paths:
        try:
            aggregate_multiple(label_paths, Path(args.aggregate_output))
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
