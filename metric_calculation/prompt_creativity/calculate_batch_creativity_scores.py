"""
Calculate Creativity Scores for Batch-Generated Visual Prompts (Lite Version)
Compatible with flat-list JSON structure.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from creativity_evaluator import (
    MusicToImageCreativityEvaluator,
    CreativityMetrics
)

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

    args = parser.parse_args()

    # Create calculator
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

    # Run evaluation
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

    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
