"""
Batch Music Analysis from WAV Audio Files using Hybrid Music Analyzer

Performs comprehensive music analysis using the 3-step Neuro-Symbolic pipeline:
- Step 1: Neural Perception (Qwen2-Audio) - Qualitative analysis
- Step 2: Symbolic Extraction (Librosa DSP) - Quantitative metrics
- Step 3: Integrated Synthesis (Qwen2-Audio) - Causal analysis

Saves results to JSON with complete music analysis data.
"""

import asyncio
import json
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from hybrid_music_analyzer import create_hybrid_analyzer

# Qwen2-Audio imports
try:
    QWEN_AUDIO_AVAILABLE = True
except ImportError:
    QWEN_AUDIO_AVAILABLE = False
    logging.warning("Qwen2-Audio not installed. Install with: pip install transformers torch librosa")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchMusicAnalyzer:
    """Analyze music for all audio samples in batch using Hybrid Music Analyzer."""

    def __init__(
        self,
        audio_dir: str,
        output_dir: str = "music_analysis_results",
        metadata_file: Optional[str] = None,
        use_qwen_audio: bool = False,
        qwen_device: str = "cuda"
    ):
        """
        Initialize batch music analyzer.

        Args:
            audio_dir: Path to directory containing WAV files
            output_dir: Directory to save results
            metadata_file: Optional path to metadata.json (generated from extract_audio_dataset.py)
            use_qwen_audio: Whether to use Qwen2-Audio for music analysis (required for hybrid analysis)
            qwen_device: Device for Qwen2-Audio ('cuda' or 'cpu')
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_qwen_audio = use_qwen_audio and QWEN_AUDIO_AVAILABLE

        # Collect WAV files
        logger.info(f"Loading WAV files from {audio_dir}")
        self.wav_files = sorted([f for f in self.audio_dir.glob("*.wav")])
        logger.info(f"Found {len(self.wav_files)} audio samples")

        # Load metadata if available
        self.metadata_dict = {}
        if metadata_file and Path(metadata_file).exists():
            logger.info(f"Loading metadata from {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                self.metadata_dict = {m['filename']: m for m in metadata_list}
                logger.info(f"Loaded metadata for {len(self.metadata_dict)} files")
        else:
            logger.warning("No metadata file provided. Using filename-based metadata only.")

        # Initialize Hybrid Music Analyzer if requested
        self.hybrid_analyzer = None
        if self.use_qwen_audio:
            try:
                self.hybrid_analyzer = create_hybrid_analyzer(device=qwen_device)
                if self.hybrid_analyzer is None:
                    raise RuntimeError("Failed to create hybrid analyzer")
                logger.info("✅ Hybrid Music Analyzer (Neuro-Symbolic) initialized for 3-step music analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Hybrid Music Analyzer: {e}. Continuing without hybrid audio analysis.")
                self.use_qwen_audio = False

        # Results storage
        self.results = {
            "metadata": {
                "audio_directory": str(self.audio_dir),
                "metadata_file": metadata_file if metadata_file else "None",
                "analysis_timestamp": datetime.now().isoformat(),
                "total_samples": len(self.wav_files),
                "hybrid_music_analyzer_enabled": self.use_qwen_audio,
                "hybrid_analyzer_device": qwen_device if self.use_qwen_audio else "N/A",
                "analysis_framework": "Hybrid Music Analysis (Neuro-Symbolic)" if self.use_qwen_audio else "Baseline (No Hybrid Analysis)",
                "pipeline_steps": ["Step 1: Neural Perception (Qwen2-Audio)", "Step 2: Symbolic Extraction (Librosa)", "Step 3: Integrated Synthesis (Qwen2-Audio)"] if self.use_qwen_audio else [],
                "sample_rate": 16000
            },
            "samples": []
        }

    def _load_wav_file(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file.

        Args:
            wav_path: Path to WAV file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = sf.read(str(wav_path))
        return np.asarray(audio, dtype=np.float32), sr

    def _serialize_librosa_metrics(self, metrics) -> Dict[str, Any]:
        """
        Convert LibrosaMetrics dataclass to JSON-serializable dictionary.

        Args:
            metrics: LibrosaMetrics object

        Returns:
            Dictionary with serialized metrics
        """
        return {
            "tempo": float(metrics.tempo),
            "key": str(metrics.key),
            "tonality": str(metrics.tonality),
            "time_signature": str(metrics.time_signature),
            "spectral_centroid": float(metrics.spectral_centroid),
            "spectral_bandwidth": float(metrics.spectral_bandwidth),
            "zero_crossing_rate": float(metrics.zero_crossing_rate),
            "rmse": float(metrics.rmse),
            "harmonic_content": float(metrics.harmonic_content),
            "melody_contour": str(metrics.melody_contour)
        }

    def _get_metadata_for_file(self, wav_filename: str) -> Dict[str, Any]:
        """
        Get metadata for a WAV file from the metadata dictionary or extract from filename.

        Args:
            wav_filename: Name of WAV file (with extension)

        Returns:
            Dictionary with metadata
        """
        if wav_filename in self.metadata_dict:
            return self.metadata_dict[wav_filename]

        # Fallback: extract from filename
        # Format: {index}_{artist}_{track}_{slice}.wav
        parts = wav_filename.replace('.wav', '').split('_')
        return {
            "filename": wav_filename,
            "artist_name": parts[1] if len(parts) > 1 else "Unknown",
            "track_name": parts[2] if len(parts) > 2 else "Unknown",
            "slice_position": parts[3] if len(parts) > 3 else "Unknown",
            "sample_rate": 16000
        }

    async def analyze_single_sample(
        self,
        sample_idx: int,
        wav_path: Path
    ) -> Dict[str, Any]:
        """
        Perform comprehensive music analysis on a single audio sample.

        Args:
            sample_idx: Index of the sample
            wav_path: Path to WAV file

        Returns:
            Dictionary with music analysis results
        """
        try:
            logger.info(f"Processing sample {sample_idx + 1}/{len(self.wav_files)}")

            # Load audio
            audio, sr = self._load_wav_file(wav_path)

            # Get metadata
            metadata = self._get_metadata_for_file(wav_path.name)

            sample_data = {
                "sample_idx": sample_idx,
                "filename": wav_path.name,
                "filepath": str(wav_path),
                "metadata": metadata,
                "hybrid_analysis": None
            }

            # Get Hybrid Music Analysis if enabled (3-step Neuro-Symbolic Pipeline)
            if self.use_qwen_audio and self.hybrid_analyzer:
                logger.info(f"  Executing 3-step Hybrid Music Analysis (Neuro-Symbolic)...")
                hybrid_result = self.hybrid_analyzer.analyze_music(
                    audio=audio,
                    sr=sr,
                    wav_path=str(wav_path)
                )
                if hybrid_result.get("status") == "success":
                    sample_data["hybrid_analysis"] = {
                        "step1_perception": hybrid_result["pipeline"]["step1_perception"],
                        "step2_metrics": self._serialize_librosa_metrics(hybrid_result["pipeline"]["step2_metrics"]),
                        "step3_synthesis": hybrid_result["pipeline"]["step3_synthesis"],
                        "timestamp": hybrid_result["timestamp"]
                    }
                    logger.info(f"  ✓ Hybrid analysis complete (all 3 steps)")
                else:
                    logger.warning(f"  ⚠️ Hybrid analysis failed: {hybrid_result.get('error')}")
            else:
                if not self.use_qwen_audio:
                    logger.info(f"  Hybrid analysis disabled (use --qwen-audio to enable)")

            logger.info(f"  ✓ Sample {sample_idx} analysis complete")
            return sample_data

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            return {
                "sample_idx": sample_idx,
                "filename": wav_path.name,
                "error": str(e),
                "hybrid_analysis": None
            }

    async def analyze_all_samples(
        self,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze all (or specified) samples.

        Args:
            sample_indices: List of sample indices to process (None = all)

        Returns:
            Results dictionary with all analyses
        """
        if sample_indices is None:
            sample_indices = list(range(len(self.wav_files)))

        logger.info(f"Starting batch analysis for {len(sample_indices)} samples")
        if self.use_qwen_audio:
            logger.info(f"Using Hybrid Music Analyzer with 3-step Neuro-Symbolic pipeline")
        else:
            logger.info(f"Baseline analysis mode (no hybrid analyzer)")

        for idx in sample_indices:
            try:
                wav_path = self.wav_files[idx]

                sample_result = await self.analyze_single_sample(idx, wav_path)
                self.results["samples"].append(sample_result)

                # Save checkpoint every 10 samples
                if (len(self.results["samples"]) % 10) == 0:
                    self._save_results(checkpoint=True)
                    logger.info(f"Checkpoint: {len(self.results['samples'])} samples saved")

            except KeyboardInterrupt:
                logger.warning("Analysis interrupted by user")
                self._save_results(checkpoint=True)
                raise
            except Exception as e:
                logger.error(f"Critical error at sample {idx}: {e}")
                continue

        return self.results

    def _save_results(self, checkpoint: bool = False) -> Path:
        """
        Save results to JSON file.

        Args:
            checkpoint: If True, save as checkpoint file

        Returns:
            Path to saved file
        """
        if checkpoint:
            filename = f"analysis_checkpoint_{len(self.results['samples']):03d}.json"
        else:
            filename = f"music_analysis_complete.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved {len(self.results['samples'])} samples to {filepath}")
        return filepath

    def save_results(self) -> Path:
        """Save final results."""
        return self._save_results(checkpoint=False)

    def save_analysis_summary(self) -> Path:
        """
        Save summary statistics about music analysis.

        Returns:
            Path to summary file
        """
        summary = {
            "total_samples": len(self.results["samples"]),
            "analysis_timestamp": self.results["metadata"]["analysis_timestamp"],
            "hybrid_analyzer_enabled": self.results["metadata"]["hybrid_music_analyzer_enabled"],
            "samples_with_errors": sum(1 for s in self.results["samples"] if "error" in s),
            "samples_with_hybrid_analysis": sum(
                1 for s in self.results["samples"] if s.get("hybrid_analysis") is not None
            ),
            "sample_summary": []
        }

        for sample in self.results["samples"]:
            sample_summary = {
                "sample_idx": sample["sample_idx"],
                "filename": sample["filename"],
                "has_hybrid_analysis": sample.get("hybrid_analysis") is not None
            }

            # Add metrics if available
            if sample.get("hybrid_analysis"):
                metrics = sample["hybrid_analysis"].get("step2_metrics", {})
                sample_summary["key"] = metrics.get("key", "N/A")
                sample_summary["tonality"] = metrics.get("tonality", "N/A")
                sample_summary["tempo"] = metrics.get("tempo", "N/A")
                sample_summary["harmonic_content"] = metrics.get("harmonic_content", "N/A")

            summary["sample_summary"].append(sample_summary)

        filepath = self.output_dir / "analysis_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved analysis summary to {filepath}")
        return filepath

    def print_summary(self):
        """Print summary of analysis results."""
        total_samples = len(self.results["samples"])
        successful_samples = sum(
            1 for s in self.results["samples"]
            if "error" not in s
        )
        samples_with_hybrid = sum(
            1 for s in self.results["samples"]
            if s.get("hybrid_analysis") is not None
        )

        print("\n" + "=" * 70)
        print("BATCH MUSIC ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total samples processed: {total_samples}")
        print(f"Successful analyses: {successful_samples}")
        print(f"Failed samples: {total_samples - successful_samples}")
        print(f"Samples with hybrid analysis: {samples_with_hybrid}")

        if self.use_qwen_audio and samples_with_hybrid > 0:
            print(f"\nHybrid Analysis Pipeline (3-Step Neuro-Symbolic):")
            print(f"  ✓ Step 1: Neural Perception (Qwen2-Audio) - Qualitative analysis")
            print(f"  ✓ Step 2: Symbolic Extraction (Librosa) - Quantitative metrics")
            print(f"  ✓ Step 3: Integrated Synthesis (Qwen2-Audio) - Causal analysis")

        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Files created:")
        print(f"  - music_analysis_complete.json (complete analysis data)")
        print(f"  - analysis_summary.json (statistics and metrics)")
        print(f"  - music_analysis.log (detailed execution log)")
        print("=" * 70 + "\n")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch music analysis from audio WAV files using Hybrid Music Analyzer"
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Path to directory containing WAV files"
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Path to metadata.json file (optional, will use filename parsing as fallback)"
    )
    parser.add_argument(
        "--output-dir",
        default="music_analysis_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting sample index"
    )
    parser.add_argument(
        "--qwen-audio",
        action="store_true",
        help="Enable Qwen2-Audio model for hybrid music analysis (requires CUDA GPU)"
    )
    parser.add_argument(
        "--qwen-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Qwen2-Audio model (cuda recommended for performance)"
    )

    args = parser.parse_args()

    # Create analyzer and discover WAV files
    analyzer = BatchMusicAnalyzer(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        use_qwen_audio=args.qwen_audio,
        qwen_device=args.qwen_device
    )

    total_samples = len(analyzer.wav_files)
    logger.info(f"Found {total_samples} WAV files in {args.audio_dir}")

    # Determine sample indices
    if args.samples is None:
        sample_indices = list(range(args.start_idx, total_samples))
    else:
        sample_indices = list(
            range(args.start_idx, min(args.start_idx + args.samples, total_samples))
        )

    logger.info(f"Processing {len(sample_indices)} samples out of {total_samples}")

    try:
        await analyzer.analyze_all_samples(sample_indices)
        analyzer.save_results()
        analyzer.save_analysis_summary()
        analyzer.print_summary()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
