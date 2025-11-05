"""
Local Dataset Loader for Parquet Files
로컬 parquet 파일을 읽어서 HuggingFace datasets format으로 변환
"""

import os
from pathlib import Path
from datasets import Dataset, concatenate_datasets
import pandas as pd
import glob
from typing import List, Optional, Dict
import hashlib

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False

class LocalDatasetLoader:
    """로컬 parquet 파일 로더"""

    def __init__(self, base_path: str = "~/haerae_dataset", cache_dir: str = "/tmp/dataset_cache"):
        self.base_path = Path(base_path).expanduser()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_distributed(self) -> bool:
        """DDP가 초기화되었는지 확인"""
        return DIST_AVAILABLE and dist.is_initialized()

    def _get_rank(self) -> int:
        """현재 rank 반환 (DDP가 아니면 0)"""
        if self._is_distributed():
            return dist.get_rank()
        return 0

    def _barrier(self):
        """DDP barrier (모든 rank 동기화)"""
        if self._is_distributed():
            dist.barrier()

    def _get_cache_path(self, dataset_name: str, max_samples: Optional[int] = None) -> Path:
        """캐시 경로 생성"""
        # 데이터셋 이름과 max_samples로 고유 해시 생성
        cache_key = f"{dataset_name}_{max_samples}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        return self.cache_dir / f"{dataset_name}_{cache_hash}"

    def load_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        text_column: str = 'text',
        preprocessing_fn: Optional[callable] = None
    ) -> Dataset:
        """
        로컬 parquet 파일 로드 (DDP-aware)

        Args:
            dataset_name: 데이터셋 이름 (예: "KOREAN-WEBTEXT")
            max_samples: 최대 샘플 수
            text_column: 텍스트 컬럼 이름
            preprocessing_fn: 전처리 함수

        Returns:
            HuggingFace Dataset
        """
        rank = self._get_rank()
        cache_path = self._get_cache_path(dataset_name, max_samples)

        # 캐시가 이미 있으면 로드
        if cache_path.exists():
            if rank == 0:
                print(f"[Rank {rank}] Loading from cache: {cache_path}")
            dataset = Dataset.load_from_disk(str(cache_path))
            self._barrier()
            return dataset

        # Rank 0만 데이터 로드
        if rank == 0:
            dataset_path = self.base_path / dataset_name / "default" / "train"

            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            # 모든 parquet 파일 찾기
            parquet_files = sorted(glob.glob(str(dataset_path / "*.parquet")))

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files in: {dataset_path}")

            print(f"[Rank 0] Found {len(parquet_files)} parquet files in {dataset_name}")

            # 파일들을 순차적으로 읽으면서 max_samples 제한
            dfs = []
            total_samples = 0

            for pq_file in parquet_files:
                df = pd.read_parquet(pq_file)

                if max_samples and total_samples + len(df) > max_samples:
                    # 남은 샘플 수만큼만 가져오기
                    remaining = max_samples - total_samples
                    df = df.head(remaining)
                    dfs.append(df)
                    break

                dfs.append(df)
                total_samples += len(df)

                if max_samples and total_samples >= max_samples:
                    break

            # 모든 DataFrame 합치기
            combined_df = pd.concat(dfs, ignore_index=True)

            # 전처리 함수 적용
            if preprocessing_fn:
                combined_df = preprocessing_fn(combined_df)

            # text 컬럼 확인 및 변환
            if text_column not in combined_df.columns:
                # 자동으로 text 컬럼 찾기
                possible_columns = ['text', 'content', 'question', 'Question', 'response']
                for col in possible_columns:
                    if col in combined_df.columns:
                        text_column = col
                        break
                else:
                    raise ValueError(f"No text column found in {dataset_name}")

            # 'text' 컬럼으로 통일
            if text_column != 'text':
                combined_df['text'] = combined_df[text_column]

            # HuggingFace Dataset으로 변환
            dataset = Dataset.from_pandas(combined_df[['text']])

            print(f"[Rank 0] Loaded {len(dataset)} samples from {dataset_name}")

            # 캐시에 저장
            dataset.save_to_disk(str(cache_path))
            print(f"[Rank 0] Saved to cache: {cache_path}")

        # Barrier: Rank 0이 끝날 때까지 대기
        self._barrier()

        # 모든 rank가 캐시에서 로드
        if rank != 0:
            dataset = Dataset.load_from_disk(str(cache_path))

        self._barrier()
        return dataset

    def load_kosimpleeval(
        self,
        subset: str = "all",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        KoSimpleEval 데이터셋 로드 (여러 서브셋 결합)

        Args:
            subset: "all" 또는 특정 서브셋 이름
            max_samples: 최대 샘플 수

        Returns:
            HuggingFace Dataset
        """
        base_path = self.base_path / "KoSimpleEval"

        if not base_path.exists():
            raise FileNotFoundError(f"KoSimpleEval not found: {base_path}")

        # 모든 서브셋 폴더 찾기
        subsets = [d.name for d in base_path.iterdir() if d.is_dir()]

        print(f"Found KoSimpleEval subsets: {subsets}")

        dfs = []
        for subset_name in subsets:
            subset_path = base_path / subset_name / "test"
            parquet_file = subset_path / "0000.parquet"

            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                # question 컬럼에서 텍스트 추출
                if 'question' in df.columns:
                    df['text'] = df['question'].str.split('###').str[1]
                    df['text'] = df['text'].fillna(df['question'])
                    dfs.append(df[['text']])
                elif 'text' in df.columns:
                    # 이미 'text' 컬럼이 있는 경우
                    dfs.append(df[['text']])

        # 합치기
        if not dfs:
            print(f"Warning: No valid data found in KoSimpleEval subsets")
            # 빈 데이터셋 생성
            combined_df = pd.DataFrame({'text': []})
        else:
            combined_df = pd.concat(dfs, ignore_index=True)

        if max_samples and len(combined_df) > 0:
            combined_df = combined_df.head(max_samples)

        dataset = Dataset.from_pandas(combined_df)

        print(f"Loaded {len(dataset)} samples from KoSimpleEval")

        return dataset

    def load_cot_dataset(
        self,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        HAE-RAE-COT 데이터셋 로드 (Question + CoT_Rationale) - DDP-aware
        """
        rank = self._get_rank()
        cache_path = self._get_cache_path("HAE-RAE-COT", max_samples)

        # 캐시가 있으면 로드
        if cache_path.exists():
            dataset = Dataset.load_from_disk(str(cache_path))
            self._barrier()
            return dataset

        # Rank 0만 로드
        if rank == 0:
            dataset_path = self.base_path / "HAE-RAE-COT" / "default" / "train"
            parquet_files = sorted(glob.glob(str(dataset_path / "*.parquet")))

            dfs = []
            total_samples = 0

            for pq_file in parquet_files:
                df = pd.read_parquet(pq_file)

                # Question + CoT_Rationale 결합
                df['text'] = df['Question'].astype(str) + " " + df['CoT_Rationale'].astype(str)

                if max_samples and total_samples + len(df) > max_samples:
                    remaining = max_samples - total_samples
                    df = df.head(remaining)
                    dfs.append(df[['text']])
                    break

                dfs.append(df[['text']])
                total_samples += len(df)

                if max_samples and total_samples >= max_samples:
                    break

            combined_df = pd.concat(dfs, ignore_index=True)
            dataset = Dataset.from_pandas(combined_df)

            print(f"[Rank 0] Loaded {len(dataset)} samples from HAE-RAE-COT")
            dataset.save_to_disk(str(cache_path))

        self._barrier()
        if rank != 0:
            dataset = Dataset.load_from_disk(str(cache_path))
        self._barrier()

        return dataset

    def load_math_dataset(
        self,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        HR-Instruct-Math 데이터셋 로드 (instruction + response) - DDP-aware
        """
        rank = self._get_rank()
        cache_path = self._get_cache_path("HR-Instruct-Math", max_samples)

        if cache_path.exists():
            dataset = Dataset.load_from_disk(str(cache_path))
            self._barrier()
            return dataset

        if rank == 0:
            dataset_path = self.base_path / "HR-Instruct-Math" / "default" / "train"
            parquet_file = dataset_path / "0000.parquet"

            df = pd.read_parquet(parquet_file)

            # instruction + response 결합
            df['text'] = df['instruction'].astype(str) + " " + df['response'].astype(str)

            if max_samples:
                df = df.head(max_samples)

            dataset = Dataset.from_pandas(df[['text']])

            print(f"[Rank 0] Loaded {len(dataset)} samples from HR-Instruct-Math")
            dataset.save_to_disk(str(cache_path))

        self._barrier()
        if rank != 0:
            dataset = Dataset.load_from_disk(str(cache_path))
        self._barrier()

        return dataset

    def load_feedback_dataset(
        self,
        min_score: int = 5,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        K2-Feedback 데이터셋 로드 (높은 점수만) - DDP-aware
        """
        rank = self._get_rank()
        cache_path = self._get_cache_path(f"K2-Feedback-score{min_score}", max_samples)

        if cache_path.exists():
            dataset = Dataset.load_from_disk(str(cache_path))
            self._barrier()
            return dataset

        if rank == 0:
            dataset_path = self.base_path / "K2-Feedback" / "default" / "train"
            parquet_files = sorted(glob.glob(str(dataset_path / "*.parquet")))

            dfs = []
            for pq_file in parquet_files:
                df = pd.read_parquet(pq_file)

                # Score 5인 것만 필터링
                if 'score' in df.columns:
                    df = df[df['score'] == min_score]

                # response 컬럼 사용
                if 'response' in df.columns:
                    df['text'] = df['response'].astype(str)

                dfs.append(df[['text']])

            combined_df = pd.concat(dfs, ignore_index=True)

            if max_samples:
                combined_df = combined_df.head(max_samples)

            dataset = Dataset.from_pandas(combined_df)

            print(f"[Rank 0] Loaded {len(dataset)} samples from K2-Feedback (score={min_score})")
            dataset.save_to_disk(str(cache_path))

        self._barrier()
        if rank != 0:
            dataset = Dataset.load_from_disk(str(cache_path))
        self._barrier()

        return dataset

    def load_mixed_dataset(
        self,
        dataset_configs: List[Dict]
    ) -> Dataset:
        """
        여러 데이터셋을 섞어서 로드

        Args:
            dataset_configs: [
                {"name": "KOREAN-WEBTEXT", "max_samples": 100000, "weight": 0.5},
                {"name": "KOREAN-SyntheticText", "max_samples": 50000, "weight": 0.3}
            ]

        Returns:
            Combined Dataset
        """
        datasets = []

        for config in dataset_configs:
            name = config['name']
            max_samples = config.get('max_samples')

            if name == "KoSimpleEval":
                ds = self.load_kosimpleeval(max_samples=max_samples)
            elif name == "HAE-RAE-COT":
                ds = self.load_cot_dataset(max_samples=max_samples)
            elif name == "HR-Instruct-Math":
                ds = self.load_math_dataset(max_samples=max_samples)
            elif name == "K2-Feedback":
                ds = self.load_feedback_dataset(max_samples=max_samples)
            else:
                ds = self.load_dataset(name, max_samples=max_samples)

            datasets.append(ds)

        # 데이터셋 결합
        combined = concatenate_datasets(datasets)

        # 셔플
        combined = combined.shuffle(seed=42)

        print(f"Combined dataset: {len(combined)} total samples")

        return combined
