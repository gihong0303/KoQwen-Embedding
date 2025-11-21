#!/usr/bin/env python3
"""Stage 1: New Token Input Embeddings"""
import os
import argparse
from transformers import set_seed
from base_trainer import BaseEmbeddingTrainer, cleanup_distributed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        trainer = BaseEmbeddingTrainer("stage1", args.config, model_path=None)
        trainer.train()
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
