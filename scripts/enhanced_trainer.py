"""
Enhanced Trainer with Curriculum Learning Support
Extends BaseEmbeddingTrainer with curriculum functionality
"""

import sys
from pathlib import Path
from datasets import Dataset, load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_trainer import BaseEmbeddingTrainer
from utils.curriculum_dataset import apply_curriculum

# Import local dataset loader
import importlib.util
spec = importlib.util.spec_from_file_location("local_dataset_loader", PROJECT_ROOT / "utils" / "local_dataset_loader.py")
local_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_dataset_module)
LocalDatasetLoader = local_dataset_module.LocalDatasetLoader


class EnhancedEmbeddingTrainer(BaseEmbeddingTrainer):
    """
    Enhanced trainer with curriculum learning support
    """

    def prepare_data(self):
        """Override to add curriculum learning"""
        self.log("\n" + "=" * 80)
        self.log("Data Preparation (Enhanced with Curriculum)")
        self.log("=" * 80)

        dataset_cfg = self.stage_config['dataset']

        # Load dataset (supports mixed datasets from HuggingFace)
        if dataset_cfg.get('mixed'):
            self.log(f"\n🗂️  Loading mixed datasets:")
            datasets_list = []

            for ds_config in dataset_cfg['mixed']:
                ds_name = ds_config['name']
                max_samples = ds_config.get('max_samples')

                self.log(f"     • {ds_name}: {max_samples:,} samples")

                # Load based on source
                source = ds_config.get('source', 'huggingface')

                if source == 'huggingface':
                    # Load from HuggingFace
                    config_name = ds_config.get('config', None)
                    if config_name:
                        ds = load_dataset(
                            ds_name,
                            config_name,
                            split=ds_config.get('split', 'train'),
                            streaming=False
                        )
                    else:
                        ds = load_dataset(
                            ds_name,
                            split=ds_config.get('split', 'train'),
                            streaming=False
                        )

                    if max_samples:
                        ds = ds.select(range(min(max_samples, len(ds))))

                    # Convert to 'text' column if not exists
                    if 'text' not in ds.column_names:
                        # Handle NLI datasets (premise + hypothesis)
                        if 'premise' in ds.column_names and 'hypothesis' in ds.column_names:
                            ds = ds.map(
                                lambda x: {'text': f"{x['premise']} {x['hypothesis']}"},
                                remove_columns=[c for c in ds.column_names if c != 'text'],
                                num_proc=4
                            )
                            self.log(f"       → Converted NLI format to text")
                        # Handle other common formats
                        elif 'sentence' in ds.column_names:
                            ds = ds.rename_column('sentence', 'text')
                        elif 'content' in ds.column_names:
                            ds = ds.rename_column('content', 'text')
                        elif 'question' in ds.column_names:
                            ds = ds.rename_column('question', 'text')

                elif source == 'local':
                    # Load from local path
                    loader = LocalDatasetLoader(base_path=ds_config.get('local_path', '~/haerae_dataset'))

                    if "KOREAN-WEBTEXT" in ds_name:
                        ds = loader.load_dataset("KOREAN-WEBTEXT", max_samples=max_samples)
                    elif "KOREAN-SyntheticText" in ds_name:
                        ds = loader.load_dataset("KOREAN-SyntheticText", max_samples=max_samples)
                    elif "KoSimpleEval" in ds_name:
                        ds = loader.load_kosimpleeval(max_samples=max_samples)
                    elif "HAE-RAE-COT" in ds_name:
                        ds = loader.load_cot_dataset(max_samples=max_samples)
                    elif "HR-Instruct-Math" in ds_name:
                        ds = loader.load_math_dataset(max_samples=max_samples)
                    else:
                        ds = loader.load_dataset(ds_name, max_samples=max_samples)

                datasets_list.append(ds)

            # Concatenate all datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets_list)

        elif dataset_cfg.get('local', False):
            # Single local dataset
            self.log(f"\n🗂️  Loading local dataset: {dataset_cfg['name']}")
            loader = LocalDatasetLoader(base_path=dataset_cfg.get('local_path', '~/haerae_dataset'))

            if dataset_cfg['name'] == "KoSimpleEval":
                dataset = loader.load_kosimpleeval(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "HAE-RAE-COT":
                dataset = loader.load_cot_dataset(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "HR-Instruct-Math":
                dataset = loader.load_math_dataset(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "K2-Feedback":
                dataset = loader.load_feedback_dataset(
                    min_score=dataset_cfg.get('min_score', 5),
                    max_samples=dataset_cfg.get('max_samples')
                )
            else:
                dataset = loader.load_dataset(dataset_cfg['name'], max_samples=dataset_cfg.get('max_samples'))

        else:
            # Single HuggingFace dataset
            self.log(f"\n🤗 Loading HuggingFace dataset: {dataset_cfg['name']}")
            dataset = load_dataset(
                dataset_cfg['name'],
                split='train',
                streaming=dataset_cfg.get('streaming', False)
            )

            if dataset_cfg.get('max_samples'):
                dataset = dataset.select(range(min(dataset_cfg['max_samples'], len(dataset))))

        # Apply curriculum learning if enabled
        curriculum_cfg = self.stage_config.get('curriculum', {})

        if curriculum_cfg.get('enabled', False):
            self.log(f"\n🎓 Applying Curriculum Learning:")
            self.log(f"   Mode: {curriculum_cfg['mode']}")
            self.log(f"   Priority weight: {curriculum_cfg['priority_weight']}")

            # Determine stage number from stage name
            stage_num = int(self.stage_name[-1]) if self.stage_name[-1].isdigit() else 4

            dataset = apply_curriculum(
                dataset=dataset,
                tokenizer=self.tokenizer,
                difficulty_categories_path=curriculum_cfg['difficulty_categories_path'],
                stage=stage_num,
                max_samples=None,  # Already limited above
                priority_weight=curriculum_cfg['priority_weight']
            )

            self.log(f"   ✓ Curriculum applied, dataset size: {len(dataset):,}")

        # Continue with standard data loader setup
        def collate_fn(examples):
            # Extract texts with validation
            texts = []
            for ex in examples:
                text = ex.get('text', '')
                # Ensure text is a valid string
                if text is None:
                    text = ''
                elif not isinstance(text, str):
                    text = str(text)
                # Filter out empty strings
                if text.strip():
                    texts.append(text)
                else:
                    texts.append('[EMPTY]')  # Placeholder for empty texts

            encodings = self.tokenizer(
                texts,
                max_length=self.stage_config['training']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }

        from torch.utils.data import DataLoader, DistributedSampler

        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.stage_config['training']['batch_size'],
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        self.log(f"\n✓ DataLoader ready")
        self.log(f"  Batch/GPU: {self.stage_config['training']['batch_size']}")
        self.log(f"  Total batches: {len(self.train_dataloader)}")
