#!/usr/bin/env python3
"""Stage 4: Full Vocabulary Harmonization"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import set_seed
from scripts.enhanced_trainer import EnhancedEmbeddingTrainer
from scripts.base_trainer import cleanup_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Stage 3 model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        trainer = EnhancedEmbeddingTrainer("stage4", args.config, model_path=args.model_path)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
