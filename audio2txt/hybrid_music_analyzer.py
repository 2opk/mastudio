"""
Hybrid Music Analysis Framework (Neuro-Symbolic Approach)

Integrates Qwen2-Audio (Neural Perception) with Librosa (Symbolic Extraction)
for comprehensive, data-grounded, narrative-driven music analysis.

3-Step Pipeline:
1. Step 1: Neural Perception (Qualitative) - Qwen2-Audio initial perceptual analysis
2. Step 2: Symbolic Extraction (Quantitative) - Librosa hard data extraction
3. Step 3: Integrated Synthesis (Deep Analysis) - Qwen2-Audio comprehensive report
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    import librosa
    import torch
    QWEN_AUDIO_AVAILABLE = True
except ImportError:
    QWEN_AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LibrosaMetrics:
    """Container for Librosa DSP analysis results"""
    tempo: float  # BPM
    beat_frames: np.ndarray  # Beat frame positions
    chroma: np.ndarray  # Chromagram for key detection
    key: str  # Detected key/scale
    tonality: str  # Major/Minor
    spectral_centroid: float  # Timbre brightness (Hz)
    spectral_bandwidth: float  # Spectral spread
    zero_crossing_rate: float  # High-frequency content
    rmse: float  # Root Mean Square Energy (loudness)
    mel_spectrogram: np.ndarray  # Mel-scale spectrogram
    time_signature: str  # Estimated time signature
    harmonic_content: float  # Harmonic vs percussive ratio
    melody_contour: str  # Melodic direction description


class LibrosaSymbolicExtractor:
    """Step 2: Symbolic Extraction - Quantitative Music Analysis using Librosa DSP"""

    def __init__(self, sr: int = 16000):
        """
        Initialize the symbolic extractor.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr
        logger.info(f"Initialized LibrosaSymbolicExtractor (sr={sr})")

    def extract_metrics(self, audio: np.ndarray, sr: int = None) -> LibrosaMetrics:
        """
        Extract comprehensive quantitative metrics from audio.

        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (uses self.sr if not provided)

        Returns:
            LibrosaMetrics object with all extracted features
        """
        if sr is None:
            sr = self.sr

        try:
            logger.info("Extracting symbolic (quantitative) features from audio...")

            # 1. Rhythm Analysis - Beat tracking and tempo
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo)
            logger.info(f"  ✓ Tempo: {tempo:.1f} BPM")

            # 2. Tonality Analysis - Chromagram and key detection
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key = self._detect_key(chroma_mean)
            tonality = self._detect_tonality(chroma_mean)
            logger.info(f"  ✓ Key: {key} ({tonality})")

            # 3. Timbre/Texture Analysis - Spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            logger.info(f"  ✓ Timbre - Spectral Centroid: {spectral_centroid:.1f} Hz, ZCR: {zcr:.4f}")

            # 4. Dynamics Analysis - Energy/Loudness
            rms_value = librosa.feature.rms(y=audio)
            rmse = float(np.mean(rms_value))
            logger.info(f"  ✓ Dynamics - RMSE Energy: {rmse:.4f}")

            # 5. Mel-Spectrogram for texture analysis
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            logger.info(f"  ✓ Mel-Spectrogram computed ({mel_spec_db.shape})")

            # 6. Time Signature estimation
            time_sig = self._estimate_time_signature(beat_frames)

            # 7. Harmonic content (harmonic vs percussive separation)
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive))))
            logger.info(f"  ✓ Harmonic ratio: {harmonic_ratio:.3f}")

            # 8. Melody contour
            contour = self._analyze_melody_contour(mel_spec_db)

            metrics = LibrosaMetrics(
                tempo=tempo,
                beat_frames=beat_frames,
                chroma=chroma,
                key=key,
                tonality=tonality,
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
                zero_crossing_rate=zcr,
                rmse=rmse,
                mel_spectrogram=mel_spec_db,
                time_signature=time_sig,
                harmonic_content=harmonic_ratio,
                melody_contour=contour
            )

            logger.info("✅ Symbolic extraction complete")
            return metrics

        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            raise

    def _detect_key(self, chroma_mean: np.ndarray) -> str:
        """Detect key from chromagram"""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_idx = np.argmax(chroma_mean)
        return note_names[key_idx]

    def _detect_tonality(self, chroma_mean: np.ndarray) -> str:
        """Detect major/minor tonality"""
        # Simplified: use energy distribution
        # Major keys have higher energy in 3rd, 7th positions
        major_positions = [4, 11]  # E and B in chroma
        minor_positions = [0, 9]  # C and A in chroma
        major_energy = np.sum(chroma_mean[major_positions])
        minor_energy = np.sum(chroma_mean[minor_positions])
        return "Major" if major_energy > minor_energy else "Minor"

    def _estimate_time_signature(self, beat_frames: np.ndarray) -> str:
        """Estimate time signature from beat frames"""
        if len(beat_frames) < 2:
            return "4/4"
        # Simplified: most common is 4/4
        # Could be enhanced with onset detection
        return "4/4"

    def _analyze_melody_contour(self, mel_spec_db: np.ndarray) -> str:
        """Analyze melodic contour (rising, falling, undulating)"""
        if mel_spec_db.shape[1] < 2:
            return "Unknown"

        # Get centroid of mel spectrogram over time
        contour = np.argmax(mel_spec_db, axis=0)
        if len(contour) < 10:
            return "Unknown"

        # Analyze trend
        trend = np.mean(np.diff(contour[:10])) - np.mean(np.diff(contour[-10:]))

        if trend > 5:
            return "Rising"
        elif trend < -5:
            return "Falling"
        else:
            return "Undulating"

    def format_metrics_for_prompt(self, metrics: LibrosaMetrics) -> str:
        """Format metrics as readable text for Qwen2-Audio synthesis"""
        return f"""
## Technical Features (from DSP Analysis)
- **Key/Tonality**: {metrics.key} {metrics.tonality}
- **Tempo**: {metrics.tempo:.1f} BPM
- **Time Signature**: {metrics.time_signature}
- **Spectral Centroid**: {metrics.spectral_centroid:.1f} Hz (Timbre Brightness)
- **Spectral Bandwidth**: {metrics.spectral_bandwidth:.1f} Hz (Spectral Spread)
- **Zero Crossing Rate**: {metrics.zero_crossing_rate:.4f} (High-frequency Content)
- **Dynamic Intensity (RMSE)**: {metrics.rmse:.4f} (Loudness Level)
- **Harmonic Content**: {metrics.harmonic_content:.1%} (Harmonic vs Percussive Ratio)
- **Melody Contour**: {metrics.melody_contour}
"""


class HybridMusicAnalyzer:
    """
    Hybrid Music Analysis Framework combining Neural and Symbolic approaches.

    3-Step Pipeline:
    1. Neural Perception (Qwen2-Audio) - Qualitative analysis
    2. Symbolic Extraction (Librosa) - Quantitative analysis
    3. Integrated Synthesis (Qwen2-Audio) - Deep analysis combining both
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-Audio-7B-Instruct", device: str = "cuda"):
        """
        Initialize the hybrid analyzer.

        Args:
            model_name: HuggingFace model identifier for Qwen2-Audio
            device: Device to run model on ('cuda' or 'cpu')
        """
        if not QWEN_AUDIO_AVAILABLE:
            raise RuntimeError(
                "Qwen2-Audio is not installed. Install with: "
                "pip install transformers torch librosa"
            )

        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.symbolic_extractor = LibrosaSymbolicExtractor(sr=16000)
        self._load_model()
        logger.info("✅ HybridMusicAnalyzer initialized")

    def _load_model(self):
        """Load Qwen2-Audio model and processor"""
        try:
            logger.info(f"Loading Qwen2-Audio model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device,
                trust_remote_code=True
            ).eval()
            logger.info("✅ Qwen2-Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen2-Audio model: {e}")
            raise

    def analyze_music(self, audio: np.ndarray, sr: int, wav_path: str) -> Dict[str, Any]:
        """
        Execute the full 3-step Hybrid Music Analysis Pipeline.

        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            wav_path: Path to audio file (for reference)

        Returns:
            Comprehensive analysis result with all 3 steps
        """
        try:
            logger.info("=" * 70)
            logger.info("🎵 HYBRID MUSIC ANALYSIS PIPELINE (Neuro-Symbolic)")
            logger.info("=" * 70)

            # Step 1: Neural Perception
            logger.info("\n[STEP 1] 🧠 Neural Perception (Qualitative Analysis)")
            logger.info("-" * 70)
            step1_perception = self._step1_neural_perception(audio, sr, wav_path)

            # Step 2: Symbolic Extraction
            logger.info("\n[STEP 2] 📊 Symbolic Extraction (Quantitative Analysis)")
            logger.info("-" * 70)
            step2_metrics = self._step2_symbolic_extraction(audio, sr)

            # Step 3: Integrated Synthesis
            logger.info("\n[STEP 3] 🔗 Integrated Synthesis (Deep Analysis)")
            logger.info("-" * 70)
            step3_synthesis = self._step3_integrated_synthesis(
                audio, sr, wav_path, step1_perception, step2_metrics
            )

            return {
                "status": "success",
                "pipeline": {
                    "step1_perception": step1_perception,
                    "step2_metrics": step2_metrics,
                    "step3_synthesis": step3_synthesis
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "wav_path": wav_path
            }

    def _step1_neural_perception(self, audio: np.ndarray, sr: int, wav_path: str) -> Dict[str, Any]:
        """
        STEP 1: Neural Perception - Initial Qualitative Analysis using Qwen2-Audio.

        Extracts texture, atmosphere, instrumentation, and genre style.
        Constraints: Does NOT guess quantitative metrics (BPM, Key) to avoid hallucination.
        """
        try:
            logger.info("Loading audio and generating initial perceptual analysis...")

            # Prompt for Step 1: Focus on "feeling" without numerical guessing
            step1_prompt = """Analyze to this audio track. Provide a purely perceptual analysis focusing on the 'feel' and 'texture' of the sound.

Please analyze the following aspects WITHOUT guessing or estimating technical metrics like exact BPM, Key, or Time Signature:

1. **Genre & Style**: Identify the specific sub-genre and stylistic influences. What musical traditions does it draw from?

2. **Instrumentation & Timbre**: What instruments or sounds are present? Describe their sonic characteristics (e.g., 'warm analog synth', 'distorted aggressive guitar', 'smooth jazz trumpet').

3. **Vocal Characteristics**: If vocals are present, describe the vocal style, character, and lyrical mood (happy, melancholic, rebellious, etc.)

4. **Atmosphere & Imagery**: What visual scenes, emotions, or sensory experiences does this track evoke? What color would you associate with it?

5. **Structural Perception**: Describe the overall flow and evolution of the track (e.g., 'starts quiet and intimate, builds to a chaotic climax', 'sparse intro followed by full orchestration')

6. **Energy & Dynamic Feel**: How would you describe the overall energy level? Is it static or does it shift throughout?"""

            conversation = [
                {
                    'role': 'system',
                    'content': 'You are an expert music critic with a trained ear for musical nuance. Focus on the perceptual and emotional qualities of music rather than technical metrics.'
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": step1_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = self.processor(
                text=text,
                audios=[audio],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=256)

            # Extract only the generated response (skip the prompt template)
            response_text = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True
            )[0]

            # Remove the prompt template from the response
            # The response contains the full conversation, we need just the assistant's response
            if 'assistant' in response_text:
                perception_text = response_text.split('assistant')[-1].strip()
            else:
                perception_text = response_text

            logger.info("✅ Step 1 perception analysis complete")
            return {
                "perception": perception_text,
                "focus": "Qualitative - Texture, Atmosphere, Instrumentation, Genre"
            }

        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            return {
                "perception": f"Error: {str(e)}",
                "focus": "Qualitative"
            }

    def _step2_symbolic_extraction(self, audio: np.ndarray, sr: int) -> LibrosaMetrics:
        """
        STEP 2: Symbolic Extraction - Quantitative Analysis using Librosa DSP.

        Extracts hard data: Tempo, Key, Timbre, Dynamics, etc.
        """
        return self.symbolic_extractor.extract_metrics(audio, sr)

    def _step3_integrated_synthesis(
        self,
        audio: np.ndarray,
        sr: int,
        wav_path: str,
        step1_perception: Dict[str, Any],
        step2_metrics: LibrosaMetrics
    ) -> Dict[str, Any]:
        """
        STEP 3: Integrated Synthesis - Deep Analysis combining both approaches.

        Uses Librosa data as evidence to validate and ground the qualitative analysis.
        Explains causality: "How do these technical features create this specific musical atmosphere?"
        """
        try:
            logger.info("Synthesizing comprehensive analysis using both neural and symbolic data...")

            # Format Step 2 metrics for the prompt
            metrics_text = self.symbolic_extractor.format_metrics_for_prompt(step2_metrics)

            # Prompt for Step 3: Synthesis with data grounding
            step3_prompt = f"""Based on your analyzation of the audio and the provided technical data below, generate a comprehensive and deeply analytical musical analysis report.

## INPUT DATA

### 1. Initial Perceptual Notes (from Qualitative Analysis):
{step1_perception['perception']}

### 2. Technical Features (from DSP Analysis):
{metrics_text}

## SYNTHESIS INSTRUCTIONS

Your task is to synthesize these inputs into a structured, professional musicology report. Your goal is to explain **HOW** the technical features create the perceived atmosphere and emotional impact.

**Grounding Principle**: Use the technical data (Key, Tempo, Spectral Centroid, etc.) to support and validate your descriptions. For example:
- "The high Spectral Centroid value of {step2_metrics.spectral_centroid:.0f} Hz correlates with the bright, piercing synthesizer timbre observed in the intro"
- "The energetic {step2_metrics.tempo:.0f} BPM drives the chaotic energy, preventing the listener from settling into comfort"

**Conflict Resolution**: If DSP data appears to contradict the perceived mood (e.g., Major key but melancholic mood), explain this artistic juxtaposition as a deliberate compositional choice.

**Causal Analysis**: For each observation, explain the mechanism:
- How does the {step2_metrics.tonality} tonality shape listener expectations?
- How does the {step2_metrics.harmonic_content:.0%} harmonic content create the perceived texture?
- How does the {step2_metrics.melody_contour} melody contour drive the emotional narrative?

## FINAL OUTPUT STRUCTURE

Generate a report with these sections:

### 📋 Executive Summary
A concise overview of the track's identity, core musical appeal, and emotional signature. (2-3 sentences)

### 🎯 Rhythmic & Harmonic Architecture
Deep dive into how the {step2_metrics.tempo:.0f} BPM and {step2_metrics.key} {step2_metrics.tonality} construct the harmonic and rhythmic foundation. Explain how the {step2_metrics.time_signature} time signature organizes the pulse.

### 🎨 Timbral & Textural Landscape
Detailed instrument analysis grounded in spectral features:
- How does the {step2_metrics.spectral_centroid:.0f} Hz Spectral Centroid explain the timbral brightness/darkness?
- How does the Zero Crossing Rate of {step2_metrics.zero_crossing_rate:.4f} reflect high-frequency content?
- What role does the {step2_metrics.harmonic_content:.0%} harmonic ratio play in the texture?

### 📈 Emotional & Narrative Evolution
Explain how the track evolves emotionally, referencing:
- The {step2_metrics.melody_contour} melody contour and its emotional trajectory
- Dynamic changes reflected in RMSE energy variations
- How harmonic progression builds meaning across the track structure

### 💡 Deep Insights
Any additional musical insights that bridge the perceptual and technical analysis. How do all these elements combine to create the listener's experience?"""

            conversation = [
                {
                    'role': 'system',
                    'content': 'You are a lead musicologist and data analyst. Your expertise is synthesizing perceptual observations with technical music theory and signal processing data. Provide insightful, grounded analysis that explains the causality between technical features and musical experience.'
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": step3_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = self.processor(
                text=text,
                audios=[audio],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                # Use max_new_tokens instead of max_length to allow the model to handle long prompts
                # This lets the input fit within model's context window
                generate_ids = self.model.generate(**inputs, max_new_tokens=512)

            # Extract only the generated response (skip the prompt template)
            synthesis_full = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True
            )[0]

            # Remove the prompt template from the response
            # The response contains the full conversation, we need just the assistant's response
            if 'assistant' in synthesis_full:
                synthesis_text = synthesis_full.split('assistant')[-1].strip()
            else:
                synthesis_text = synthesis_full

            logger.info("✅ Step 3 synthesis complete")
            return {
                "synthesis": synthesis_text,
                "focus": "Integrated - Causal Analysis & Deep Insights"
            }

        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            return {
                "synthesis": f"Error: {str(e)}",
                "focus": "Integrated"
            }


def create_hybrid_analyzer(device: str = "cuda") -> HybridMusicAnalyzer:
    """Factory function to create HybridMusicAnalyzer with error handling"""
    try:
        return HybridMusicAnalyzer(device=device)
    except Exception as e:
        logger.error(f"Failed to create HybridMusicAnalyzer: {e}")
        return None
