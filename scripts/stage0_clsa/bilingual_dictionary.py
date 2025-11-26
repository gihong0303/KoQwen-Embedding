#!/usr/bin/env python3
"""
Bilingual Dictionary Extraction for CLSA
한국어-영어-중국어 토큰 매핑 자동 생성
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import logging

from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BilingualDictionaryExtractor:
    """
    Cross-lingual token mapping 생성

    Method:
    1. Extract Korean tokens from expanded tokenizer
    2. Use base model embeddings to find nearest English/Chinese tokens
    3. Filter by confidence score
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B",  # Lightweight Qwen for token extraction
        korean_tokenizer_path: str = None,
        device: str = "cuda"
    ):
        self.device = device
        logger.info(f"Loading base model: {base_model_name}")

        # Base tokenizer (original Qwen)
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        # Korean-expanded tokenizer
        if korean_tokenizer_path:
            self.korean_tokenizer = AutoTokenizer.from_pretrained(
                korean_tokenizer_path,
                trust_remote_code=True
            )
        else:
            self.korean_tokenizer = self.base_tokenizer

        # Lightweight model for semantic matching
        self.model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device).eval()

        logger.info("Model loaded successfully")

    def is_korean_token(self, token: str) -> bool:
        """Check if token contains Korean characters"""
        korean_chars = set('가-힣ㄱ-ㅎㅏ-ㅣ')
        return any(char in korean_chars for char in token)

    def is_english_token(self, token: str) -> bool:
        """Check if token is primarily English"""
        # Remove special chars
        clean = token.strip().replace('▁', '').replace('Ġ', '')
        if not clean:
            return False
        # Check if mostly ascii letters
        letters = [c for c in clean if c.isalpha()]
        if not letters:
            return False
        english_count = sum(1 for c in letters if ord(c) < 128)
        return english_count / len(letters) > 0.8

    def is_chinese_token(self, token: str) -> bool:
        """Check if token contains Chinese characters"""
        chinese_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3400, 0x4DBF),   # CJK Extension A
        ]
        return any(
            any(start <= ord(char) <= end for start, end in chinese_ranges)
            for char in token
        )

    def extract_language_tokens(self) -> Dict[str, Set[str]]:
        """Extract tokens by language from base tokenizer"""
        logger.info("Extracting language-specific tokens...")

        lang_tokens = {
            'korean': set(),
            'english': set(),
            'chinese': set()
        }

        base_vocab = self.base_tokenizer.get_vocab()

        for token in tqdm(base_vocab.keys(), desc="Analyzing base vocab"):
            if self.is_korean_token(token):
                lang_tokens['korean'].add(token)
            elif self.is_english_token(token):
                lang_tokens['english'].add(token)
            elif self.is_chinese_token(token):
                lang_tokens['chinese'].add(token)

        logger.info(f"Base vocab analysis:")
        logger.info(f"  Korean tokens: {len(lang_tokens['korean']):,}")
        logger.info(f"  English tokens: {len(lang_tokens['english']):,}")
        logger.info(f"  Chinese tokens: {len(lang_tokens['chinese']):,}")

        return lang_tokens

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a single token"""
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight[token_id]
        return emb.cpu().float()

    def find_cross_lingual_anchors(
        self,
        new_korean_tokens: List[str],
        top_k: int = 5,
        min_similarity: float = 0.3,
        batch_size: int = 512
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Find English/Chinese anchor tokens for each Korean token (BATCH OPTIMIZED)

        Args:
            new_korean_tokens: List of new Korean tokens to map
            top_k: Number of anchors per language
            min_similarity: Minimum cosine similarity threshold
            batch_size: Batch size for GPU processing

        Returns:
            {
                'korean_token': [
                    ('english_token', 'en', similarity),
                    ('chinese_token', 'zh', similarity),
                    ...
                ]
            }
        """
        logger.info(f"Finding cross-lingual anchors for {len(new_korean_tokens)} tokens (BATCH MODE)...")

        # Get language tokens from base vocab
        lang_tokens = self.extract_language_tokens()

        # Build embedding matrices for English and Chinese
        en_token_list = list(lang_tokens['english'])[:10000]  # Limit for memory
        zh_token_list = list(lang_tokens['chinese'])[:10000]

        logger.info(f"Building embedding matrices...")
        logger.info(f"  English: {len(en_token_list):,} tokens")
        logger.info(f"  Chinese: {len(zh_token_list):,} tokens")

        # Get all embeddings from model at once (GPU optimized)
        embed_weight = self.model.get_input_embeddings().weight

        # English embeddings (batch)
        en_token_ids = torch.tensor([self.base_tokenizer.convert_tokens_to_ids(t) for t in en_token_list], device=self.device)
        with torch.no_grad():
            en_embeddings = embed_weight[en_token_ids].float()
            en_embeddings = torch.nn.functional.normalize(en_embeddings, p=2, dim=1)
        logger.info(f"  English embeddings ready: {en_embeddings.shape}")

        # Chinese embeddings (batch, handle empty case)
        zh_embeddings = None
        if zh_token_list:
            zh_token_ids = torch.tensor([self.base_tokenizer.convert_tokens_to_ids(t) for t in zh_token_list], device=self.device)
            with torch.no_grad():
                zh_embeddings = embed_weight[zh_token_ids].float()
                zh_embeddings = torch.nn.functional.normalize(zh_embeddings, p=2, dim=1)
            logger.info(f"  Chinese embeddings ready: {zh_embeddings.shape}")
        else:
            logger.info("  No Chinese tokens found in base vocabulary, skipping Chinese anchors")

        # Precompute Korean token embeddings in batches
        logger.info(f"Computing Korean token embeddings in batches of {batch_size}...")

        base_vocab = self.base_tokenizer.get_vocab()
        anchors_map = {}

        # Process in batches
        num_batches = (len(new_korean_tokens) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(new_korean_tokens))
            batch_tokens = new_korean_tokens[start_idx:end_idx]

            # Compute embeddings for this batch
            batch_embeddings = []
            valid_tokens = []

            for ko_token in batch_tokens:
                if ko_token in base_vocab:
                    ko_id = self.base_tokenizer.convert_tokens_to_ids(ko_token)
                    with torch.no_grad():
                        ko_emb = embed_weight[ko_id].float()
                else:
                    # Subword averaging
                    subtokens = self.base_tokenizer.tokenize(ko_token)
                    if not subtokens:
                        continue
                    subtoken_ids = self.base_tokenizer.convert_tokens_to_ids(subtokens)
                    subtoken_ids = torch.tensor(subtoken_ids, device=self.device)
                    with torch.no_grad():
                        ko_emb = embed_weight[subtoken_ids].float().mean(dim=0)

                batch_embeddings.append(ko_emb)
                valid_tokens.append(ko_token)

            if not batch_embeddings:
                continue

            # Stack and normalize
            batch_embeddings = torch.stack(batch_embeddings)  # [batch, dim]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            # Compute similarities with English (batch matrix multiplication)
            with torch.no_grad():
                en_similarities = batch_embeddings @ en_embeddings.T  # [batch, en_vocab]
                en_topk_values, en_topk_indices = torch.topk(en_similarities, k=min(top_k, len(en_token_list)), dim=1)

            # Compute similarities with Chinese (if available)
            zh_topk_values, zh_topk_indices = None, None
            if zh_embeddings is not None:
                with torch.no_grad():
                    zh_similarities = batch_embeddings @ zh_embeddings.T
                    zh_topk_values, zh_topk_indices = torch.topk(zh_similarities, k=min(top_k, len(zh_token_list)), dim=1)

            # Collect results
            for i, ko_token in enumerate(valid_tokens):
                anchors = []

                # English anchors
                for j in range(en_topk_values.shape[1]):
                    sim = en_topk_values[i, j].item()
                    if sim >= min_similarity:
                        idx = en_topk_indices[i, j].item()
                        anchors.append((en_token_list[idx], 'en', sim))

                # Chinese anchors
                if zh_topk_values is not None:
                    for j in range(zh_topk_values.shape[1]):
                        sim = zh_topk_values[i, j].item()
                        if sim >= min_similarity:
                            idx = zh_topk_indices[i, j].item()
                            anchors.append((zh_token_list[idx], 'zh', sim))

                if anchors:
                    anchors_map[ko_token] = anchors

        logger.info(f"Found anchors for {len(anchors_map):,}/{len(new_korean_tokens):,} tokens")

        return anchors_map

    def save_bilingual_dictionary(
        self,
        anchors_map: Dict,
        output_path: str
    ):
        """Save bilingual dictionary to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable = {
            token: [
                {'anchor': anchor, 'lang': lang, 'similarity': float(sim)}
                for anchor, lang, sim in anchors
            ]
            for token, anchors in anchors_map.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved bilingual dictionary to: {output_path}")

        # Statistics
        total_anchors = sum(len(v) for v in serializable.values())
        avg_anchors = total_anchors / len(serializable) if serializable else 0

        logger.info(f"Statistics:")
        logger.info(f"  Total Korean tokens: {len(serializable):,}")
        logger.info(f"  Total anchors: {total_anchors:,}")
        logger.info(f"  Average anchors/token: {avg_anchors:.2f}")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract bilingual dictionary")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model for token embeddings"
    )
    parser.add_argument(
        "--korean_tokenizer",
        type=str,
        default="outputs/koqwen-expanded",
        help="Korean-expanded tokenizer path"
    )
    parser.add_argument(
        "--vocab_diff_path",
        type=str,
        default="tokenizer/vocab_diff.json",
        help="Vocabulary difference file (new Korean tokens)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bilingual_dictionary.json",
        help="Output path for bilingual dictionary"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of anchors per language"
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.3,
        help="Minimum similarity threshold"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,
        help="Maximum number of Korean tokens to process (for testing)"
    )

    args = parser.parse_args()

    # Load new Korean tokens
    logger.info(f"Loading vocabulary diff from: {args.vocab_diff_path}")
    with open(args.vocab_diff_path, 'r', encoding='utf-8') as f:
        vocab_diff = json.load(f)

    # Extract token list from vocab_diff
    if isinstance(vocab_diff, dict):
        if "safe_add_tokens" in vocab_diff:
            new_korean_tokens = vocab_diff["safe_add_tokens"][:args.max_tokens]
        elif "tokens" in vocab_diff:
            new_korean_tokens = vocab_diff["tokens"][:args.max_tokens]
        else:
            new_korean_tokens = [k for k in vocab_diff.keys() if k != "statistics"][:args.max_tokens]
    else:
        new_korean_tokens = vocab_diff[:args.max_tokens]

    logger.info(f"Processing {len(new_korean_tokens):,} new Korean tokens")

    # Extract bilingual dictionary
    extractor = BilingualDictionaryExtractor(
        base_model_name=args.base_model,
        korean_tokenizer_path=args.korean_tokenizer
    )

    anchors_map = extractor.find_cross_lingual_anchors(
        new_korean_tokens=new_korean_tokens,
        top_k=args.top_k,
        min_similarity=args.min_similarity
    )

    # Save
    extractor.save_bilingual_dictionary(
        anchors_map=anchors_map,
        output_path=args.output
    )

    logger.info("✅ Bilingual dictionary extraction complete!")


if __name__ == "__main__":
    main()
