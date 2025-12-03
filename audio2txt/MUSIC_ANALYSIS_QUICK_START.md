# Music Analysis Script - Quick Start Guide

**Script**: `generate_text_from_audio_batch.py`
**Purpose**: Batch music analysis using Hybrid Music Analyzer (3-step Neuro-Symbolic pipeline)
**Status**: ✅ Production Ready

---

## 5-Minute Overview

This script analyzes your audio files using a sophisticated 3-step pipeline:

1. **Step 1 - Neural Perception** (Qwen2-Audio): Qualitative analysis of music characteristics
2. **Step 2 - Symbolic Extraction** (Librosa): Quantitative metrics extraction (tempo, key, tonality, spectral features, etc.)
3. **Step 3 - Integrated Synthesis** (Qwen2-Audio): Causal analysis grounded in quantitative metrics

Results are saved as JSON with complete analysis data for each audio sample.

---

## Installation

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Ensure pip is up to date
pip install --upgrade pip
```

### Install Dependencies (One-Time Setup)

```bash
# Install required packages
pip install transformers torch librosa soundfile

# Optional: Use conda for easier installation
conda install -c conda-forge transformers pytorch librosa soundfile
```

### Verify Installation

```bash
python3 -c "
from transformers import Qwen2AudioForConditionalGeneration
import librosa
import torch
print('✅ All dependencies installed successfully')
"
```

---

## Quick Start Examples

### 1. Test with 2 Samples (Recommended First Step)

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "path/to/audio/folder" \
    --samples 2 \
    --qwen-audio
```

**What happens:**
- Loads 2 WAV files from your audio directory
- Runs 3-step hybrid analysis on each (~60-75 seconds per sample with GPU)
- Saves results to `music_analysis_results/` directory
- Total time: ~2-3 minutes

---

### 2. Full Batch Processing (Production)

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --metadata-file "dataset/metadata.json" \
    --output-dir "results" \
    --samples 100 \
    --qwen-audio \
    --qwen-device cuda
```

**What happens:**
- Analyzes 100 audio samples
- Uses metadata from JSON file for enriched output
- Saves results to `results/` directory
- Saves checkpoint every 10 samples
- Total time: ~1-2 hours (with GPU)

---

### 3. Baseline Analysis (No Hybrid, Fast)

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --samples 5
```

**What happens:**
- Quick analysis without hybrid analyzer
- Useful for testing directory structure
- No GPU required
- Total time: < 1 minute

---

### 4. Resume from Checkpoint

If processing is interrupted, you can resume:

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --start-idx 50 \
    --qwen-audio
```

Processes samples starting from index 50.

---

## Command Reference

### Required Arguments

```
--audio-dir PATH
```
Directory containing WAV audio files to analyze.

### Optional Arguments

```
--metadata-file PATH
```
Path to metadata.json file (generated from `extract_audio_dataset.py`). Enriches results with additional information.

```
--output-dir PATH
```
Where to save results. Default: `music_analysis_results/`

```
--samples N
```
Number of samples to process. Default: all files in directory.

```
--start-idx N
```
Starting sample index. Default: 0. Useful for resuming interrupted analysis.

```
--qwen-audio
```
Enable hybrid music analyzer with Qwen2-Audio. Recommended for production analysis. Without this flag, no hybrid analysis is performed.

```
--qwen-device DEVICE
```
GPU or CPU for Qwen2-Audio. Options: `cuda` (default), `cpu`. Use `cuda` for best performance.

---

## Output Structure

### Directory Layout

```
music_analysis_results/
├── music_analysis_complete.json      # Final results (all samples)
├── analysis_summary.json             # Summary statistics
├── analysis_checkpoint_010.json      # Checkpoint after 10 samples
├── analysis_checkpoint_020.json      # Checkpoint after 20 samples
└── music_analysis.log                # Detailed execution log
```

### JSON Structure (Sample)

```json
{
  "metadata": {
    "audio_directory": "dataset/audio",
    "analysis_timestamp": "2025-12-03T10:30:45.123456",
    "total_samples": 100,
    "hybrid_music_analyzer_enabled": true,
    "hybrid_analyzer_device": "cuda",
    "analysis_framework": "Hybrid Music Analysis (Neuro-Symbolic)",
    "pipeline_steps": [
      "Step 1: Neural Perception (Qwen2-Audio)",
      "Step 2: Symbolic Extraction (Librosa)",
      "Step 3: Integrated Synthesis (Qwen2-Audio)"
    ]
  },
  "samples": [
    {
      "sample_idx": 0,
      "filename": "0000_artist_track_0-30.wav",
      "filepath": "dataset/audio/0000_artist_track_0-30.wav",
      "metadata": {
        "artist_name": "artist",
        "track_name": "track",
        "slice_position": "0-30"
      },
      "hybrid_analysis": {
        "step1_perception": {
          "perception": "This track features a melancholic atmosphere with..."
        },
        "step2_metrics": {
          "tempo": 120.5,
          "key": "C",
          "tonality": "Minor",
          "time_signature": "4/4",
          "spectral_centroid": 3500.2,
          "spectral_bandwidth": 2100.5,
          "zero_crossing_rate": 0.0456,
          "rmse": 0.0234,
          "harmonic_content": 0.653,
          "melody_contour": "Undulating"
        },
        "step3_synthesis": {
          "synthesis": "The C minor tonality establishes a melancholic foundation..."
        },
        "timestamp": "2025-12-03T10:30:46.123456"
      }
    }
  ]
}
```

### Music Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| **tempo** | 0-300 BPM | Beats per minute |
| **key** | C-B | Musical key (root note) |
| **tonality** | Major/Minor | Scale type |
| **time_signature** | e.g., 4/4 | Beats per measure |
| **spectral_centroid** | 0-22050 Hz | Brightness of sound |
| **spectral_bandwidth** | 0-22050 Hz | Spread of frequencies |
| **zero_crossing_rate** | 0-1 | Noisiness of signal |
| **rmse** | 0-1 | Overall loudness |
| **harmonic_content** | 0-1 | Tonal vs percussive (1=pure tone) |
| **melody_contour** | Ascending/Descending/Undulating | Direction of pitch movement |

---

## Monitoring Progress

### Check Real-Time Progress

```bash
# Watch log file as it processes
tail -f music_analysis.log

# Count completed samples
grep "✓ Sample" music_analysis.log | wc -l

# Check for errors
grep "ERROR\|WARNING" music_analysis.log
```

### Check Intermediate Results

```bash
# See current checkpoint
python3 -c "
import json
with open('music_analysis_results/analysis_checkpoint_010.json') as f:
    data = json.load(f)
    print(f'Samples processed: {len(data[\"samples\"])}')
"
```

---

## Performance Expectations

### Processing Time (With GPU)

| Scenario | Time per Sample | Total for 100 Samples |
|----------|-----------------|----------------------|
| Hybrid Analysis (GPU) | 60-75 seconds | 1-2 hours |
| Hybrid Analysis (CPU) | 3-5 minutes | 5-8 hours |
| Baseline (no hybrid) | 1-2 seconds | 1-2 minutes |

### Resource Requirements

```
GPU (CUDA):
  - VRAM: ~11 GB (for Qwen2-Audio model)
  - Disk: ~10 GB (for model download)
  - CPU: Moderate usage

CPU-Only:
  - RAM: ~4 GB
  - Speed: 3-5x slower than GPU
  - Not recommended for large batches
```

---

## Common Tasks

### Process All Samples in Directory

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --qwen-audio
```

### Process with Metadata Enrichment

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --metadata-file "dataset/metadata.json" \
    --output-dir "analyzed_results" \
    --qwen-audio
```

### CPU-Only Processing (No GPU)

```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --samples 10 \
    --qwen-audio \
    --qwen-device cpu
```

### Extract Key Metrics to CSV

```python
import json
import csv

# Load results
with open('music_analysis_results/music_analysis_complete.json') as f:
    data = json.load(f)

# Extract metrics to CSV
with open('metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Tempo', 'Key', 'Tonality', 'Harmonic Content'])

    for sample in data['samples']:
        if sample.get('hybrid_analysis'):
            metrics = sample['hybrid_analysis']['step2_metrics']
            writer.writerow([
                sample['filename'],
                metrics['tempo'],
                metrics['key'],
                metrics['tonality'],
                metrics['harmonic_content']
            ])
```

---

## Troubleshooting

### Issue: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers torch librosa soundfile
```

### Issue: CUDA out of memory

**Solution 1 - Use CPU:**
```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --samples 5 \
    --qwen-audio \
    --qwen-device cpu
```

**Solution 2 - Reduce batch size:**
```bash
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio/subset" \
    --qwen-audio
```

### Issue: Model download very slow

This is normal on first run (~5-10 GB download). Subsequent runs use cached model:
- Allow 10-30 minutes for first download
- Ensure stable internet connection
- Model is cached locally after first download

### Issue: "No WAV files found"

Check your audio directory:
```bash
# List WAV files
ls -la dataset/audio/*.wav

# Ensure directory exists
test -d dataset/audio && echo "Directory exists" || echo "Directory not found"
```

### Issue: Analysis interrupted, want to resume

Find the latest checkpoint and resume:
```bash
# List checkpoints
ls -la music_analysis_results/analysis_checkpoint_*.json

# Resume from sample 50
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --start-idx 50 \
    --qwen-audio
```

---

## Advanced: Cluster Deployment

### SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --output=music_analysis_%j.log

# Load environment
module load python/3.10
module load cuda/11.8

# Install dependencies
pip install transformers torch librosa soundfile

# Run analysis
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --metadata-file "dataset/metadata.json" \
    --output-dir "results" \
    --qwen-audio \
    --qwen-device cuda
```

### Submit Job

```bash
sbatch slurm_job.sh

# Monitor progress
tail -f music_analysis_*.log
```

---

## Next Steps

1. ✅ **Install dependencies**: `pip install transformers torch librosa soundfile`
2. ✅ **Test with 2 samples**: `python3 generate_text_from_audio_batch.py --audio-dir "path/to/audio" --samples 2 --qwen-audio`
3. ✅ **Check results**: `cat music_analysis_results/music_analysis_complete.json | head -50`
4. ✅ **Process full batch**: Add more samples and run on GPU for best performance
5. ✅ **Extract metrics**: Use results for downstream analysis or visualization

---

## Questions?

### Where are the results?

Results are in `music_analysis_results/` directory (or custom `--output-dir`):
- `music_analysis_complete.json` - All analysis results
- `analysis_summary.json` - Summary statistics
- `music_analysis.log` - Execution log

### How long does it take?

With GPU: ~60-75 seconds per sample (~1-2 hours for 100 samples)
With CPU: ~3-5 minutes per sample (~5-8 hours for 100 samples)

### What if processing fails?

Checkpoints are saved every 10 samples. Resume from `--start-idx N` to continue.

### Can I use without GPU?

Yes, but add `--qwen-device cpu`. Processing will be 3-5x slower.

### Do I need metadata file?

No, it's optional. Script extracts metadata from filenames if not provided.

---

## Summary

```bash
# Quick start (2 samples)
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --samples 2 \
    --qwen-audio

# Production (full batch)
python3 generate_text_from_audio_batch.py \
    --audio-dir "dataset/audio" \
    --metadata-file "dataset/metadata.json" \
    --output-dir "results" \
    --qwen-audio \
    --qwen-device cuda
```

That's it! Your music analysis is ready to run. 🎵

---

**Last Updated**: December 3, 2025
**Status**: Production Ready ✅
