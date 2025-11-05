#!/usr/bin/env python3
"""Stage 3: New Token Refinement"""
import os
import argparse
from transformers import set_seed
from base_trainer import BaseEmbeddingTrainer, cleanup_distributed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        trainer = BaseEmbeddingTrainer("stage3", args.config, model_path=args.model_path)
        trainer.train()
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
