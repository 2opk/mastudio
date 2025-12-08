import os
from typing import Optional
import logging

# pip install openai 필요
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class GPTJudge:
    """
    GPT-4 Client for Creativity Evaluation.
    Acts as the 'Judge' to score prompts based on creativity principles.
    """
    def __init__(self, model: str = "gpt-4-turbo-preview", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        Args:
            model: Model to use (default: gpt-4-turbo for better reasoning)
            api_key: OpenAI API Key (optional if set in env vars)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    async def analyze(self, prompt: str) -> str:
        """
        Send evaluation prompt to GPT-4 and get the analysis.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert art critic and creativity researcher. Your task is to evaluate the creativity of visual descriptions based on music, using strict psychometric criteria (Originality, Elaboration, Alignment, Coherence)."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic output for consistent evaluation
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"GPT-4 API Error: {e}")
            return ""
