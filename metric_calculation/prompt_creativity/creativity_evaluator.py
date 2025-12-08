"""
Creativity Evaluator for Music-to-Image Visual Prompts (Theoretical Extension)

Implements AUT (Alternative Uses Test) and TTCT (Torrance Test of Creative Thinking)
metrics, extended with domain-specific mappings from Art Theory (Kandinsky, Arnheim).

The key innovation is the injection of 'Theoretical Knowledge' into the Alignment Scorer,
moving beyond simple semantic similarity to structural correspondence checking.

Extension based on: Goes et al. (2023) - ICCC-2023 Paper
"""

import re
from typing import Dict, Any, List, Set
from dataclasses import dataclass
from llm_client import GPTJudge


@dataclass
class CreativityMetrics:
    """Container for creativity evaluation results"""
    originality: float  # 1-5: Novelty + Theoretical Depth
    elaboration: float  # 1-5: Detail + Sensory Richness
    alignment: float    # 1-5: Theoretical Music-Visual Correspondence
    coherence: float    # 1-5: Syntactic and Semantic Unity
    overall: float      # 1-5: Weighted Score

    def to_dict(self) -> Dict[str, float]:
        return {
            'originality': self.originality,
            'elaboration': self.elaboration,
            'alignment': self.alignment,
            'coherence': self.coherence,
            'overall': self.overall
        }

class OriginalityScorer:
    """
    Scores originality by penalizing clichés and rewarding theoretical/artistic vocabulary.
    Based on Martindale's theory of 'Remote Associations'.
    """
    
    # 1. Cliché Phrases (Low Originality)
    CLICHE_PHRASES = {
        'sad': ['sad', 'crying', 'tears', 'rain', 'grey', 'gloomy'],
        'happy': ['happy', 'smile', 'sun', 'bright', 'cheerful'],
        'love': ['heart', 'love', 'kiss', 'red', 'rose'],
        'general': ['music notes', 'playing instrument', 'sound waves', 'audio visualizer']
    }

    # 2. Theoretical & Artistic Vocabulary (High Originality)
    # Words derived from Kandinsky, Arnheim, and Art History (Baroque, Pop Art, Abstract)
    THEORETICAL_LEXICON = {
        # Kandinsky (Point, Line, Plane)
        'kinetic', 'trajectory', 'intersection', 'geometric', 'planar', 'tension',
        'composition', 'vibration', 'resonance', 'dissonance', 'syncopation',
        
        # Arnheim (Visual Perception)
        'equilibrium', 'balance', 'asymmetry', 'weight', 'vector', 'gradient',
        'centrality', 'distortion', 'perspective', 'depth',
        
        # Art Styles & Techniques (High-level concepts)
        'chiaroscuro', 'impasto', 'pointillism', 'juxtaposition', 'surreal',
        'ethereal', 'iridescent', 'luminescent', 'monochromatic', 'polychromatic',
        'fractal', 'recursive', 'tessellation', 'amorphous', 'ephemeral'
    }

    def score(self, prompt: str) -> float:
        words = set(re.findall(r'\w+', prompt.lower()))
        if not words: return 1.0

        # Calculate Penalties
        cliche_hits = 0
        for category, phrases in self.CLICHE_PHRASES.items():
            for phrase in phrases:
                if phrase in prompt.lower():
                    cliche_hits += 1
        
        # Calculate Rewards
        theory_hits = sum(1 for word in words if word in self.THEORETICAL_LEXICON)

        # Normalize logic: Start at 3.0
        # -0.5 per cliche, +0.5 per theoretical term
        score = 3.0 - (cliche_hits * 0.5) + (theory_hits * 0.5)
        
        return max(1.0, min(5.0, score))

class AlignmentScorer:
    """
    [Core Contribution]
    Scores alignment based on 'Synesthetic Translation Rules' derived from 
    Kandinsky ('Point and Line to Plane') and Arnheim ('Art and Visual Perception').
    """

    # Mapping Logic: Music Feature -> Visual Keywords
    THEORETICAL_MAPPINGS = {
        # 1. Rhythm & Articulation -> Form (Kandinsky)
        'staccato': { # Point / Impact
            'keywords': {'dot', 'point', 'speck', 'burst', 'scattered', 'particle', 'pixel', 'stipple', 'sharp'},
            'music_condition': lambda f: f.get('articulation') == 'staccato' or f.get('tempo', 0) > 130
        },
        'legato': { # Line / Flow
            'keywords': {'line', 'curve', 'flow', 'stream', 'wave', 'ribbon', 'continuous', 'smooth', 'liquid'},
            'music_condition': lambda f: f.get('articulation') == 'legato' or f.get('tempo', 0) < 80
        },

        # 2. Pitch & Key -> Space & Brightness (Kandinsky & Arnheim)
        'high_pitch_major': { # Upper / Light / Vertical
            'keywords': {'bright', 'light', 'top', 'sky', 'rising', 'vertical', 'white', 'yellow', 'ethereal'},
            'music_condition': lambda f: f.get('tonality') == 'major' or f.get('pitch_range') == 'high'
        },
        'low_pitch_minor': { # Lower / Dark / Horizontal
            'keywords': {'dark', 'heavy', 'bottom', 'ground', 'deep', 'shadow', 'horizontal', 'blue', 'black', 'massive'},
            'music_condition': lambda f: f.get('tonality') == 'minor' or f.get('pitch_range') == 'low'
        },

        # 3. Harmony & Tension -> Structure (Arnheim)
        'consonance': { # Balance / Center
            'keywords': {'balance', 'symmetry', 'center', 'circle', 'stable', 'calm', 'harmonious', 'ordered'},
            'music_condition': lambda f: f.get('harmony') == 'consonant' or f.get('mood') == 'calm'
        },
        'dissonance': { # Imbalance / Diagonal
            'keywords': {'diagonal', 'angle', 'jagged', 'clashing', 'chaos', 'tension', 'distorted', 'asymmetric'},
            'music_condition': lambda f: f.get('harmony') == 'dissonant' or f.get('mood') in ['tense', 'angry']
        },
        
        # 4. Timbre/Texture -> Visual Texture
        'complex_timbre': { # Texture / Layers
            'keywords': {'layered', 'texture', 'grain', 'rough', 'complex', 'intricate', 'detail'},
            'music_condition': lambda f: f.get('timbre') == 'complex' or f.get('instrument_count', 1) > 3
        }
    }

    def score(self, prompt: str, musical_features: Dict[str, Any]) -> float:
        prompt_text = prompt.lower()
        active_conditions = 0
        matches = 0
        
        # Check each theoretical mapping
        for rule_name, rule in self.THEORETICAL_MAPPINGS.items():
            # If the music feature condition is met
            try:
                if rule['music_condition'](musical_features):
                    active_conditions += 1
                    # Check if any corresponding visual keyword exists in prompt
                    keywords = rule['keywords']
                    if any(kw in prompt_text for kw in keywords):
                        matches += 1
                    # Bonus: Check for synonyms or semantic proximity could go here
            except Exception:
                continue # Skip if feature missing

        if active_conditions == 0:
            return 3.0 # Neutral score if no specific features detected

        # Calculate Ratio
        match_ratio = matches / active_conditions
        
        # Scale to 1-5 (Base 2.0, max 5.0)
        # Even 50% matching of theoretical rules is considered very high alignment
        score = 2.0 + (match_ratio * 3.0) 
        
        return max(1.0, min(5.0, score))

class ElaborationScorer:
    """
    Scores elaboration based on 'Sensory Modalities'.
    A creative prompt should evoke multiple senses, not just sight.
    """
    SENSORY_TERMS = {
        'visual': {'glow', 'shine', 'color', 'dim', 'bright', 'fade'},
        'tactile': {'rough', 'smooth', 'soft', 'hard', 'sharp', 'silky', 'texture'},
        'kinetic': {'moving', 'still', 'fast', 'slow', 'running', 'floating'},
        'emotional': {'sad', 'happy', 'angry', 'fear', 'joy', 'hope', 'grief'}
    }

    def score(self, prompt: str) -> float:
        words = set(re.findall(r'\w+', prompt.lower()))
        word_count = len(words)
        
        # 1. Length Factor (Logarithmic)
        length_score = min(5.0, word_count / 10.0) # Cap at 50 words approx
        
        # 2. Sensory Diversity
        senses_triggered = 0
        for sense, keywords in self.SENSORY_TERMS.items():
            if any(k in prompt.lower() for k in keywords):
                senses_triggered += 1
        
        diversity_score = (senses_triggered / 4.0) * 5.0
        
        # Combined Score
        return (length_score * 0.4) + (diversity_score * 0.6)

class CoherenceScorer:
    """
    Scores syntactic coherence. Simple heuristic for now.
    """
    def score(self, prompt: str) -> float:
        # Check basic structure
        if not prompt or len(prompt.split()) < 3:
            return 1.0
            
        # Penalize repetition
        words = re.findall(r'\w+', prompt.lower())
        unique_ratio = len(set(words)) / len(words)
        
        # Reward connectors
        connectors = {'and', 'but', 'because', 'with', 'where', 'while', 'creating', 'forming'}
        has_connectors = any(c in words for c in connectors)
        
        score = 3.0 + (unique_ratio * 1.5) + (0.5 if has_connectors else 0)
        return max(1.0, min(5.0, score))

class MusicToImageCreativityEvaluator:
    """
    Main Evaluator Class.
    Integrates the sub-scorers to output a holistic creativity profile.
    """
    def __init__(self):
        self.originality = OriginalityScorer()
        self.alignment = AlignmentScorer()
        self.elaboration = ElaborationScorer()
        self.coherence = CoherenceScorer()
        
        # Weights reflecting research focus:
        # High emphasis on Alignment (Theoretical Fit) and Originality
        self.weights = {
            'originality': 0.3,
            'alignment': 0.4, # Core contribution
            'elaboration': 0.2,
            'coherence': 0.1
        }

    def evaluate(self, prompt: str, musical_features: Dict[str, Any]) -> CreativityMetrics:
        scores = {
            'originality': self.originality.score(prompt),
            'alignment': self.alignment.score(prompt, musical_features),
            'elaboration': self.elaboration.score(prompt),
            'coherence': self.coherence.score(prompt)
        }
        
        overall = sum(scores[k] * w for k, w in self.weights.items())
        
        return CreativityMetrics(**scores, overall=overall)


class MusicToImageCreativityEvaluator:
    """
    Main Evaluator Class.
    """
    def __init__(self, use_llm: bool = False, api_key: str = None):
        # Heuristic Scorers
        self.originality = OriginalityScorer()
        self.alignment = AlignmentScorer()
        self.elaboration = ElaborationScorer()
        self.coherence = CoherenceScorer()
        
        self.use_llm = use_llm
        self.llm_client = GPTJudge(api_key=api_key) if use_llm else None
        
        self.weights = {
            'originality': 0.3,
            'alignment': 0.4,
            'elaboration': 0.2,
            'coherence': 0.1
        }

    async def evaluate(self, prompt: str, musical_features: Dict[str, Any]) -> CreativityMetrics:
        """
        Evaluate using either Heuristics or LLM based on initialization.
        """
        if self.use_llm and self.llm_client:
            return await self.evaluate_with_llm(prompt, musical_features)
        
        # Fallback to Heuristics (Rule-based)
        scores = {
            'originality': self.originality.score(prompt),
            'alignment': self.alignment.score(prompt, musical_features),
            'elaboration': self.elaboration.score(prompt),
            'coherence': self.coherence.score(prompt)
        }
        overall = sum(scores[k] * w for k, w in self.weights.items())
        return CreativityMetrics(**scores, overall=overall)

    async def evaluate_with_llm(self, prompt: str, musical_features: Dict[str, Any]) -> CreativityMetrics:
        """
        Evaluate using GPT-4 Judge.
        """
        eval_prompt = self._build_llm_eval_prompt(prompt, musical_features)
        response = await self.llm_client.analyze(eval_prompt)
        return self._parse_llm_response(response)

    def _build_llm_eval_prompt(self, prompt: str, musical_features: Dict[str, Any]) -> str:
        """
        Builds a theoretically grounded system prompt for the LLM Judge.
        It injects Kandinsky/Arnheim's theories as evaluation criteria.
        """
        
        # 1. Format Musical Features for the LLM
        feature_desc = "\n".join([f"- {k}: {v}" for k, v in musical_features.items()])
        
        # 2. Define Theoretical Rubric (Evaluation Criteria)
        rubric = """
        EVALUATION RUBRIC (Rate 1-5 for each dimension):

        1. SEMANTIC ALIGNMENT (Structural Isomorphism):
           * Criteria: Does the visual form structurally correspond to the musical features based on Kandinsky & Arnheim's theory?
           * Check for:
             - Rhythm (Staccato/Fast) -> Visual Form (Dots, Points, Scattered, Sharp)
             - Melody (Legato/Flowing) -> Visual Form (Lines, Curves, Woven, Stream)
             - Pitch (High/Major) -> Visual Space (Top, Vertical, Bright, Light)
             - Pitch (Low/Minor) -> Visual Space (Bottom, Horizontal, Heavy, Dark)
             - Harmony (Dissonant) -> Visual Tension (Diagonal, Asymmetric, Clashing)
           * Score 5: Perfect structural translation (e.g., Staccato music described as 'scattered geometric points').
           * Score 1: Complete mismatch (e.g., Fast/Staccato music described as 'calm flowing river').

        2. ORIGINALITY (Theoretical Depth):
           * Criteria: Does the prompt use sophisticated artistic vocabulary instead of clichés?
           * Penalize: Common associations (Sad='Rain', Happy='Sun', Love='Heart').
           * Reward: Theoretical terms (Chiaroscuro, Kinetic, Equilibrium, Resonance, Tessellation).
           * Score 5: Novel synthesis using art-theory terminology.
           * Score 1: Relying entirely on emotional clichés.

        3. ELABORATION (Sensory Richness):
           * Criteria: Does the prompt evoke multiple senses (Synesthesia)?
           * Check for: Visual + Tactile (Rough/Smooth) + Kinetic (Floating/Rushing) + Atmospheric (Glow/Dim).
           * Score 5: Multisensory, vivid, and highly detailed.

        4. COHERENCE (Artistic Unity):
           * Criteria: Do the elements form a unified artistic vision?
           * Score 5: Syntactically complex yet unified composition.
        """

        # 3. Construct the Final Prompt
        llm_prompt = f"""
        You are an expert Art Critic and Cognitive Scientist specializing in Cross-modal Perception.
        Your task is to evaluate a 'Visual Prompt' generated from a music track, based on strict Art Theory criteria.

        [MUSICAL FEATURES]
        {feature_desc}

        [GENERATED VISUAL PROMPT]
        '{prompt}'

        {rubric}

        [OUTPUT FORMAT]
        Provide the scores and a brief reasoning in JSON format:
        {{
            "originality": <float 1-5>,
            "elaboration": <float 1-5>,
            "alignment": <float 1-5>,
            "coherence": <float 1-5>,
            "overall": <float 1-5>,
            "reasoning": "<brief explanation based on theory>"
        }}
        """
        return llm_prompt.strip()


    def _parse_llm_response(self, response: str) -> CreativityMetrics:
        """
        Parses the LLM's JSON response. Handles potential formatting errors robustly.
        """
        import json
        import re

        try:
            # 1. Try direct JSON parsing
            data = json.loads(response)
        except json.JSONDecodeError:
            # 2. If failed, try to find JSON block within text (e.g., if LLM adds chatty text)
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")
            except Exception:
                # 3. Fallback: Regex for keys if JSON parsing fails completely
                scores = {}
                for key in ['originality', 'elaboration', 'alignment', 'coherence', 'overall']:
                    match = re.search(fr'"{key}"\s*:\s*([\d\.]+)', response, re.IGNORECASE)
                    if match:
                        scores[key] = float(match.group(1))
                    else:
                        scores[key] = 3.0 # Default fallback
                
                return CreativityMetrics(
                    originality=scores['originality'],
                    elaboration=scores['elaboration'],
                    alignment=scores['alignment'],
                    coherence=scores['coherence'],
                    overall=scores['overall']
                )

        # Map JSON data to Metrics object
        return CreativityMetrics(
            originality=float(data.get('originality', 3.0)),
            elaboration=float(data.get('elaboration', 3.0)),
            alignment=float(data.get('alignment', 3.0)),
            coherence=float(data.get('coherence', 3.0)),
            overall=float(data.get('overall', 3.0))
        )
