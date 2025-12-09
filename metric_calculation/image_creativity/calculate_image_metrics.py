"""
Calculate Image Quality Metrics: LPIPS, Vendi Score, and CLIP Score

Usage:
    python calc_metrics.py --images_dir /path/to/img_dir --prompts_file ./visual_prompt.json

Requirements:
    pip install torch torchvision lpips transformers pillow
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import asyncio
import ssl
import urllib.request

# [Imports & Dependency Checks]
try:
    import torch
    import torch.nn.functional as F
    import lpips
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision import transforms
    from transformers import CLIPProcessor, CLIPModel

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('metrics.log')]
)
logger = logging.getLogger(__name__)


class ImageMetricsCalculator:
    def __init__(self, images_dir: str, prompts_file: str, device: str = "cuda:2"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch libraries are missing.")

        self.images_dir = Path(images_dir)
        self.prompts_file = Path(prompts_file)
        self.device = device if torch.cuda.is_available() else "cpu"

        # Models
        self.lpips_fn = None
        self.inception_model = None
        self.clip_model = None
        self.clip_processor = None

        self.metrics = {
            "lpips_stats": {},
            "vendi_score": None,
            "clip_score": None,
            "timestamp": datetime.now().isoformat()
        }

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        logger.info(f"Initialized on {self.device}")
        logger.info(f"Images: {self.images_dir}")
        logger.info(f"Prompts: {self.prompts_file}")

    # --- 1. LPIPS (Diversity) ---
    def calculate_lpips(self):
        logger.info("Starting LPIPS Calculation...")

        # Initialize LPIPS
        if self.lpips_fn is None:
            self._fix_ssl()
            self.lpips_fn = lpips.LPIPS(net='alex', version='0.1').to(self.device)
            self.lpips_fn.eval()

        files = sorted(list(self.images_dir.glob("*.png")))
        if len(files) < 2:
            logger.warning("Not enough images for LPIPS (need at least 2).")
            return

        # Load & Preprocess
        tensors = []
        try:
            for f in files:
                img = Image.open(f).convert('RGB').resize((256, 256))
                t = transforms.ToTensor()(img).to(self.device) * 2 - 1
                tensors.append(t.unsqueeze(0))
        except Exception as e:
            logger.error(f"Error loading images for LPIPS: {e}")
            return

        # Pairwise Comparison
        scores = []
        with torch.no_grad():
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    scores.append(self.lpips_fn(tensors[i], tensors[j]).item())

        if scores:
            self.metrics["lpips_stats"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
            logger.info(f"LPIPS Mean: {np.mean(scores):.4f}")

    # --- 2. Vendi Score (Diversity) ---
    def calculate_vendi(self):
        logger.info("Starting Vendi Score Calculation...")

        # Initialize Inception
        if self.inception_model is None:
            weights = Inception_V3_Weights.DEFAULT
            self.inception_model = inception_v3(weights=weights).to(self.device)
            self.inception_model.fc = torch.nn.Identity()
            self.inception_model.eval()
            self.inception_preprocess = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        files = sorted(list(self.images_dir.glob("*.png")))
        if not files: return

        feats = []
        with torch.no_grad():
            for f in files:
                img = Image.open(f).convert('RGB')
                inp = self.inception_preprocess(img).unsqueeze(0).to(self.device)
                feats.append(self.inception_model(inp).cpu())

        X = torch.cat(feats, dim=0).to(self.device)
        X = F.normalize(X, p=2, dim=1)
        K = torch.mm(X, X.t()) / len(files)
        evals = torch.linalg.eigvalsh(K)
        evals = evals[evals > 1e-10]
        entropy = -torch.sum(evals * torch.log(evals))
        score = torch.exp(entropy).item()

        self.metrics["vendi_score"] = score
        logger.info(f"Vendi Score: {score:.4f}")

    # --- 3. CLIP Score (Alignment) ---
    def calculate_clip_score(self):
        logger.info("Starting CLIP Score Calculation...")

        # Load Prompts JSON
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                prompt_map = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            return

        # Initialize CLIP
        if self.clip_model is None:
            model_id = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_id)
            self.clip_model.eval()

        files = sorted(list(self.images_dir.glob("*.png")))
        scores = []

        logger.info(f"Processing {len(files)} images for CLIP Score...")

        with torch.no_grad():
            for f_path in files:
                f_name = f_path.name # e.g., "sample0.png"

                # Check if this image has a corresponding prompt
                if f_name not in prompt_map:
                    logger.debug(f"Skipping {f_name}: No prompt found in JSON.")
                    continue

                text_prompt = prompt_map[f_name]

                # Load Image
                image = Image.open(f_path).convert('RGB')

                # Process Inputs
                # truncation=True handles very long prompts by cutting them off
                inputs = self.clip_processor(
                    text=[text_prompt],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)

                # Get Embeddings
                outputs = self.clip_model(**inputs)
                img_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Normalize
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                # Cosine Similarity
                similarity = (text_embeds @ img_embeds.T).item()
                scores.append(max(similarity, 0.0))

        if scores:
            avg_score = np.mean(scores)
            self.metrics["clip_score"] = {
                "mean": float(avg_score),
                "std": float(np.std(scores)),
                "num_samples": len(scores)
            }
            logger.info(f"CLIP Score Mean: {avg_score:.4f}")
        else:
            logger.warning("No matched image-prompt pairs found. Check filenames in JSON vs Directory.")

    def _fix_ssl(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

    def save_results(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(exist_ok=True, parents=True)

        with open(out / "final_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Simple Report
        with open(out / "report.txt", 'w') as f:
            f.write(f"METRICS REPORT\n{'='*30}\n")
            if self.metrics['clip_score']:
                f.write(f"CLIP Score (Alignment): {self.metrics['clip_score']['mean']:.4f}\n")
            if self.metrics['lpips_stats']:
                f.write(f"LPIPS (Diversity):      {self.metrics['lpips_stats']['mean']:.4f}\n")
            if self.metrics['vendi_score']:
                f.write(f"Vendi (Diversity):      {self.metrics['vendi_score']:.4f}\n")

        logger.info(f"Done! Results saved to {out}")

async def main(args):
    calc = ImageMetricsCalculator(
        images_dir=args.images_dir,
        prompts_file=args.prompts_file
    )

    # Run Calculations
    calc.calculate_lpips()
    calc.calculate_vendi()
    calc.calculate_clip_score()

    calc.save_results(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to visual_prompt.json")
    parser.add_argument("--output_dir", type=str, default="metrics_result", help="Output directory")

    args = parser.parse_args()

    # Check & Install Transformers if needed
    try:
        import transformers
    except ImportError:
        print("Installing transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "transformers"])

    asyncio.run(main(args))
