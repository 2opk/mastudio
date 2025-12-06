import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import librosa
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from hybrid_music_analyzer import LibrosaSymbolicExtractor


STEP1_PROMPT = """Analyze to this audio track. Provide a purely perceptual analysis focusing on the 'feel' and 'texture' of the sound.

Please analyze the following aspects WITHOUT guessing or estimating technical metrics like exact BPM, Key, or Time Signature:

1. **Genre & Style**: Identify the specific sub-genre and stylistic influences. What musical traditions does it draw from?

2. **Instrumentation & Timbre**: What instruments or sounds are present? Describe their sonic characteristics (e.g., 'warm analog synth', 'distorted aggressive guitar', 'smooth jazz trumpet').

3. **Vocal Characteristics**: If vocals are present, describe the vocal style, character, and lyrical mood (happy, melancholic, rebellious, etc.)

4. **Atmosphere & Imagery**: What visual scenes, emotions, or sensory experiences does this track evoke? What color would you associate with it?

5. **Structural Perception**: Describe the overall flow and evolution of the track (e.g., 'starts quiet and intimate, builds to a chaotic climax', 'sparse intro followed by full orchestration')

6. **Energy & Dynamic Feel**: How would you describe the overall energy level? Is it static or does it shift throughout?"""


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = audio.astype("float32")
    stats = {
        "sr": sr,
        "peak": float(audio.max()) if audio.size else 0.0,
        "min": float(audio.min()) if audio.size else 0.0,
        "rms": float((audio ** 2).mean() ** 0.5) if audio.size else 0.0,
        "duration_s": len(audio) / sr if sr else 0,
    }
    return audio, sr, stats


def run_perception(wav_path: Path, audio, sr, processor, model, device: str, max_new_tokens: int):
    conversation = [
        {
            "role": "system",
            "content": "You are an expert music critic with a trained ear for musical nuance. Focus on the perceptual and emotional qualities of music rather than technical metrics.",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(wav_path)},
                {"type": "text", "text": STEP1_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response


def process_all(audio_dir: Path, output_dir: Path, model_name: str, device: str, max_new_tokens: int, target_sr: int):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, device_map=device, trust_remote_code=True
    ).eval()
    extractor = LibrosaSymbolicExtractor(sr=target_sr)
    output_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in sorted(audio_dir.glob("*.wav")):
        audio, sr, stats = load_audio(wav_path, target_sr)
        perception = run_perception(wav_path, audio, sr, processor, model, device, max_new_tokens)
        metrics = extractor.extract_metrics(audio, sr)
        metrics_dict = {
            "tempo": metrics.tempo,
            "key": metrics.key,
            "tonality": metrics.tonality,
            "time_signature": metrics.time_signature,
            "spectral_centroid": metrics.spectral_centroid,
            "spectral_bandwidth": metrics.spectral_bandwidth,
            "zero_crossing_rate": metrics.zero_crossing_rate,
            "rmse": metrics.rmse,
            "harmonic_content": metrics.harmonic_content,
            "melody_contour": metrics.melody_contour,
            "beat_frames": metrics.beat_frames.tolist(),
        }
        out = {
            "wav_path": str(wav_path),
            "perception": perception,
            "metrics": metrics_dict,
            "audio_stats": stats,
            "model": model_name,
        }
        out_path = output_dir / f"{wav_path.stem}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process all WAVs in audio/ to processed/ with Qwen2-Audio.")
    parser.add_argument("--audio-dir", default="audio2txt/audio", help="Directory containing WAV files")
    parser.add_argument("--output-dir", default="audio2txt/processed", help="Directory to write JSON outputs")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct", help="HuggingFace model id")
    parser.add_argument("--device", default="cuda:2", help="Device for inference (e.g., cuda:0 or cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length")
    parser.add_argument("--sr", type=int, default=16000, help="Resample audio to this sample rate")
    args = parser.parse_args()

    process_all(Path(args.audio_dir), Path(args.output_dir), args.model, args.device, args.max_new_tokens, args.sr)


if __name__ == "__main__":
    main()
