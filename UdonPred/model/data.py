import os
import pickle
import sqlite3
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset
from datasets import features as datasets_features
from datasets.load import load_from_disk
from ast import literal_eval

from .embedder import Embedder



SQLITE_IN_CLAUSE_STEP: int = 900

CacheEntry = Tuple[List[int], List[int], List[List[float]]]

Batch = Dict[str, List[Any]]


class SQLiteCache:
    """SQLite-backed cache for tokenisation and embeddings.

    Stores records keyed by `text`, with pickled blobs for `input_ids`,
    `attention_mask`, and `embedding`. Provides batched read/write helpers
    and lifecycle management.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.con = sqlite3.connect(path)
        try:
            self.con.execute("PRAGMA journal_mode=WAL;")
            self.con.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.DatabaseError:
            pass
        self._init_table()

    def _init_table(self) -> None:
        """Initialize the SQLite cache table if it doesn't exist."""
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                text TEXT PRIMARY KEY,
                input_ids BLOB NOT NULL,
                attention_mask BLOB NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )

    def get_many(
        self, texts: List[str]
    ) -> Dict[str, CacheEntry]:
        """Retrieve cached entries for multiple texts.
        
        Args:
            texts: List of text keys to retrieve.
        
        Returns:
            Dict mapping text to (input_ids, attention_mask, embedding) tuples.
        """
        if not texts:
            return {}
        out: Dict[str, CacheEntry] = {}
        for i in range(0, len(texts), SQLITE_IN_CLAUSE_STEP):
            chunk = texts[i : i + SQLITE_IN_CLAUSE_STEP]
            qmarks = ",".join(["?"] * len(chunk))
            rows = self.con.execute(
                f"SELECT text,input_ids,attention_mask,embedding FROM cache WHERE text IN ({qmarks})",
                chunk,
            ).fetchall()
            for t, b1, b2, b3 in rows:
                out[t] = (pickle.loads(b1), pickle.loads(b2), pickle.loads(b3))
        return out

    def put_many(
        self,
        items: List[Tuple[str, List[int], List[int], List[List[float]]]],
    ) -> None:
        """Insert or replace multiple cached entries.
        
        Args:
            items: List of (text, input_ids, attention_mask, embedding) tuples to cache.
        """
        if not items:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO cache(text,input_ids,attention_mask,embedding) VALUES (?,?,?,?)",
            [
                (t, pickle.dumps(ids), pickle.dumps(am), pickle.dumps(emb))
                for (t, ids, am, emb) in items
            ],
        )
        self.con.commit()

    def clear(self) -> None:
        """Clears all rows from the cache table."""
        self.con.execute("DELETE FROM cache")
        self.con.commit()

    def close(self) -> None:
        try:
            self.con.commit()
        finally:
            self.con.close()

    def cleanup_file(self) -> None:
        """Close the database and remove the cache file."""
        self.close()
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except OSError:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit."""
        self.close()

def make_tokenise_and_embed_fn(backbone_model, cache: SQLiteCache):
    """Build a batched tokenize and embed function with SQLite caching.
    
    Creates a function that tokenizes protein sequences and generates embeddings,
    using an SQLite cache to avoid redundant computation.
    
    Args:
        backbone_model: Embedder instance for tokenization and embedding.
        cache_path: Path to SQLite cache file.
    
    Returns:
        Callable: Function that processes a batch of sequences.
    """
    prefix_token_len = backbone_model.prefix_token_len

    def extract_texts(
        batch_data: Batch,
    ) -> Tuple[List[str], List[str], int]:
        """Extract unique texts and `x_` column keys from the batch.
        
        Preserves first-seen order while de-duplicating.
        
        Args:
            batch_data: Batch dictionary with sequence data.
        
        Returns:
            Tuple of (unique_texts, x_columns, batch_size).
        """
        x_columns = [k for k in batch_data.keys() if k.startswith("x_")]
        batch_size = len(batch_data[x_columns[0]]) if x_columns else 0

        texts: List[str] = []
        for col_name in x_columns:
            column_values = batch_data[col_name]
            for value in column_values:
                if value is not None:
                    texts.append(value)

        unique_texts = list(dict.fromkeys(texts))

        return unique_texts, x_columns, batch_size

    def process_missing_embeddings(
        cache: SQLiteCache,
        missing_texts: List[str],
        local_cache: Dict[str, CacheEntry],
    ) -> None:
        """Compute embeddings for missing texts and update both caches.
        
        Args:
            cache: SQLiteCache instance for persistence.
            missing_texts: List of texts without cached embeddings.
            local_cache: In-memory cache to update.
        """
        if not missing_texts:
            return

        batch_input_ids, batch_attention_mask = backbone_model.tokenise(missing_texts)
        batch_embeddings = backbone_model.embed(batch_input_ids, batch_attention_mask)

        seq_lens = [len(seq) for seq in missing_texts] 

        if isinstance(batch_input_ids, torch.Tensor):
            batch_input_ids_list = [row.tolist() for row in batch_input_ids]
        else:
            batch_input_ids_list = [list(row) for row in batch_input_ids]

        if isinstance(batch_attention_mask, torch.Tensor):
            batch_attention_mask_list = [row.tolist() for row in batch_attention_mask]
        else:
            batch_attention_mask_list = [list(row) for row in batch_attention_mask]

        trimmed_embeddings: List[List[List[float]]] = []
        if isinstance(batch_embeddings, torch.Tensor):
            for emb, seq_len in zip(batch_embeddings, seq_lens):
                end = int(seq_len) + prefix_token_len
                trimmed_embeddings.append(emb[prefix_token_len:end].tolist())
        else:
            for emb, seq_len in zip(batch_embeddings, seq_lens):
                end = int(seq_len) + prefix_token_len
                trimmed_embeddings.append(emb[prefix_token_len:end])

        cache_items = list(
            zip(
                missing_texts,
                batch_input_ids_list,
                batch_attention_mask_list,
                trimmed_embeddings,
            )
        )
        cache.put_many(cache_items)

        for text, ids, mask, emb in cache_items:
            local_cache[text] = (ids, mask, emb)

    def _tokenise_and_embed_batch(batch: Batch) -> Dict[str, List[Any]]:
        """Tokenize and embed text columns in the batch using a SQLite cache.
        
        Processes all text columns in a batch, using cached embeddings where
        available and computing embeddings for new sequences.
        
        Args:
            batch: Dictionary with sequence data.
        
        Returns:
            Dict with tokenized, masked, and embedded outputs.
        """
        unique_texts, x_columns, batch_size = extract_texts(batch)

        cached_data = cache.get_many(unique_texts)
        missing_texts = [t for t in unique_texts if t not in cached_data]

        process_missing_embeddings(cache, missing_texts, cached_data)

        output_batch: Dict[str, List[Any]] = {}
        for idx, col_name in enumerate(x_columns):
            column_values = batch[col_name]

            input_ids_col = [None] * batch_size
            attention_mask_col = [None] * batch_size
            embedding_col = [None] * batch_size

            for row_idx, value in enumerate(column_values):
                if value is None:
                    continue

                ids, mask, emb = cached_data[value]
                input_ids_col[row_idx] = ids
                attention_mask_col[row_idx] = mask
                embedding_col[row_idx] = emb

            output_batch[f"input_ids_{idx}"] = input_ids_col
            output_batch[f"attention_mask_{idx}"] = attention_mask_col
            output_batch[f"embedding_{idx}"] = embedding_col

        return output_batch

    return _tokenise_and_embed_batch


def build_datasets(
    path: str,
    backbone_model: Any,
    map_batch_rows: int = 1,
    cache_dir: str = ".cache/embeddings",
) -> DatasetDict:
    """Load a jsonl dataset, tokenize and embed sequences, and return a DatasetDict.
    
    Loads dataset from jsonl files, adds dataset name column, tokenizes/embeds
    sequences with caching, and converts to torch format.
    
    Args:
        path: Directory containing jsonl dataset files.
        backbone_model: Embedder instance for tokenization and embedding.
        map_batch_rows: Batch size for processing. Defaults to 1.
        cache_dir: Directory for embedding cache. Defaults to '.cache/embeddings'.
    
    Returns:
        DatasetDict: Dictionary with 'train', 'validation', 'test' splits.
    """

    data_files = {
        "train": f"{path}/train.jsonl",
        "test": f"{path}/test.jsonl",
        "validation": f"{path}/valid.jsonl"
    }
    data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}

    ds = load_dataset("json", data_files=data_files)

    dataset_name = os.path.basename(path)
    ds = ds.map(lambda x: {"dataset": dataset_name})

    cache_path = os.path.join(cache_dir, f"{dataset_name}.sqlite")

    cache = SQLiteCache(cache_path)
    try:
        process_fn = make_tokenise_and_embed_fn(backbone_model, cache)

        ds = ds.map(
            process_fn,
            batched=True,
            batch_size=map_batch_rows,
            desc="tokenising and embedding",
        )

        ds = ds.with_format("torch")
    finally:
        cache.close()

    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
    except OSError:
        pass
    return ds


def add_missing_features(datasets: List[DatasetDict]) -> List[DatasetDict]:
    """Ensure all datasets have the same set of features.
    
    Identifies all features across all datasets and adds missing features
    with empty arrays to datasets that don't have them.
    
    Args:
        datasets: List of DatasetDict objects.
    
    Returns:
        List[DatasetDict]: Datasets with consistent features.
    """
    feature_types: Dict[str, Any] = {}
    for dataset in datasets:
        for split in dataset.keys():
            features = dataset[split].features
            for feature_name, feature_def in features.items():
                current_def = feature_def
                while not isinstance(current_def, datasets_features.Value):
                    if hasattr(current_def, "feature"):
                        current_def = current_def.feature
                    else:
                        break  
                if hasattr(current_def, "dtype"):
                    feature_types[feature_name] = current_def.dtype

    for dataset in datasets:
        existing_features = set().union(
            *(set(dataset[s].features.keys()) for s in dataset.keys())
        )
        missing_features = set(feature_types.keys()) - existing_features

        if not missing_features:
            continue

        def add_empty_columns_batch(batch):
            batch_len = len(batch[next(iter(batch))])
            out = {}
            for feature in missing_features:
                out[feature] = [
                    np.array([], dtype=feature_types[feature]) for _ in range(batch_len)
                ]
            return out

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                add_empty_columns_batch,
                batched=True,
                desc=f"Adding missing columns to {split}",
            )

    return datasets


def combine_datasets(
    datasets: List[DatasetDict], data_config: Dict[str, Any]
) -> DatasetDict:
    """Interleave multiple datasets based on configuration fractions.
    
    Combines multiple datasets into a single DatasetDict by interleaving
    them according to specified fractions. For training splits, uses
    probabilistic interleaving based on fractions.
    
    Args:
        datasets: List of DatasetDict objects to combine.
        data_config: Data configuration with fraction settings for each dataset.
    
    Returns:
        DatasetDict: Combined dataset with interleaved splits.
    """
    combined_datasets = {}

    all_splits = set().union(*(ds.keys() for ds in datasets))
    for split in all_splits:
        split_datasets = [ds[split] for ds in datasets if split in ds.keys()]

        if split == "train":
            split_fractions = []
            valid_datasets = []

            for ds in split_datasets:
                if len(ds) == 0:
                    continue
                
                dataset_name = ds[0]["dataset"]
                fraction = float(data_config[dataset_name]["fraction"])

                if fraction > 0:
                    valid_datasets.append(ds)
                    split_fractions.append(fraction)

            split_datasets = valid_datasets
            total = sum(split_fractions)
            probabilities = [f / total for f in split_fractions] if total > 0 else None
            stopping_strategy = "first_exhausted"
        else:
            probabilities = None
            stopping_strategy = "all_exhausted"

        if not split_datasets:
            continue

        combined_datasets[split] = interleave_datasets(
            split_datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )
        combined_datasets[split] = combined_datasets[split].with_format(type="torch")

    return DatasetDict(combined_datasets)


def get_datasets(
    config: Dict[str, Any], train_filter: Optional[List[str]] = None
) -> DatasetDict:
    """Load, process/embed, and combine datasets defined in config.
    
    Main entry point for dataset loading. Loads datasets from specified paths,
    tokenizes and embeds sequences using a backbone model, applies training
    filters if specified, and combines multiple datasets with fractions.
    
    Args:
        config: Configuration dictionary with data and config keys.
        train_filter: Optional list of sample IDs to include in training. Defaults to None.
    
    Returns:
        DatasetDict: Combined dataset ready for training.
    """
    data_config = config["data"]
    backbone_config = config["config"]["backbone"]

    backbone_model = Embedder(
        backbone_name=backbone_config["name"],
        prefix_token=backbone_config["prefix_token"],
        tokeniser_type=backbone_config["tokeniser_type"],
        model_type=backbone_config["model_type"],
    )

    datasets: List[DatasetDict] = []

    for key, dataset_conf in data_config.items():
        if dataset_conf["fraction"] == 0:
            continue

        path = dataset_conf["path"]

        hf_path = os.path.join(path, "hf")

        if os.path.exists(hf_path) and os.path.isdir(hf_path):
            ds = load_from_disk(hf_path)
        else:
            ds = build_datasets(
                path,
                backbone_model,
                backbone_config.get(
                    "preprocessing_batch_size", 1
                ),  
            )
            ds.save_to_disk(hf_path)

        if train_filter is not None and "train" in ds:
            ds["train"] = ds["train"].filter(lambda x: x["id"] in train_filter)

        datasets.append(ds)

    datasets = add_missing_features(datasets)
    combined_datasets = combine_datasets(datasets, data_config)

    return combined_datasets


class DataCollator:
    """Batch collator handling padding.
    """
    def _load_tokeniser(self, backbone_name: str, tokeniser_type: str):
        """Load tokenizer for data collation.
        
        Args:
            backbone_name: Name or path of the backbone model.
            tokeniser_type: Type of tokenizer to load.
        
        Returns:
            Tokenizer instance.
        """
        transformers = import_module("transformers")
        tokeniser_class = getattr(transformers, tokeniser_type)
        return tokeniser_class.from_pretrained(
            backbone_name, use_fast=False, do_lower_case=False
        )

    def __init__(
        self,
        backbone_name: str,
        tokeniser_type: str,
        padding: float = 999,
        embedding_padding: float = 0.0,
        attention_mask_padding: int = 0,
        batch_first: bool = True,
        next_pow2: bool = False,
    ):
        """Initialize the DataCollator.
        
        Args:
            backbone_name: Name or path of the backbone model.
            tokeniser_type: Type of tokenizer to use.
            padding: Padding value for targets. Defaults to 999.
            embedding_padding: Padding value for embeddings. Defaults to 0.0.
            attention_mask_padding: Padding value for attention masks. Defaults to 0.
            batch_first: Whether to return batch dimension first. Defaults to True.
            next_pow2: Whether to pad to next power of 2. Defaults to False.
        """
        self.pad_token_id = self._load_tokeniser(
            backbone_name, tokeniser_type
        ).pad_token_id

        self.padding_value = padding
        self.embedding_padding = embedding_padding
        self.attention_mask_padding = attention_mask_padding
        self.batch_first = batch_first
        self.next_pow2 = next_pow2

    def _pad_tensor(
        self,
        tensors: List[torch.Tensor],
        target_length: int,
        padding_value: float,
    ) -> torch.Tensor:
        """Pad a list of tensors to a fixed length.
        
        Args:
            tensors: List of tensors to pad.
            target_length: Target length for padding.
            padding_value: Value to use for padding.
        
        Returns:
            torch.Tensor: Batch of padded tensors.
        """
        if self.next_pow2:
            target_length = 1 << (target_length - 1).bit_length()

        padded_tensors = []
        for tensor in tensors:
            tensor = tensor[:target_length]

            pad_needed = target_length - tensor.size(0)
            if pad_needed > 0:
                pad_shape = (pad_needed, *tensor.shape[1:])
                pad_tensor = torch.full(
                    pad_shape,
                    padding_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tensor = torch.cat([tensor, pad_tensor], dim=0)
            padded_tensors.append(tensor)

        batch_tensor = torch.stack(padded_tensors)
        return batch_tensor if self.batch_first else batch_tensor.transpose(0, 1)

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples for training.
        
        Pads sequences and embeddings to the maximum length in the batch,
        and adds indices for non-tensor values to allow access on multi-gpu setup.
        
        Args:
            batch_items: List of sample dictionaries.
        
        Returns:
            Dict: Collated batch with padded tensors and index mappings.
        """
        if not batch_items:
            return {}

        first_item = batch_items[0]
        collated_batch = {}

        for key, value in first_item.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if key.startswith("input_ids"):
                    pad_val = self.pad_token_id
                elif key.startswith("attention_mask"):
                    pad_val = self.attention_mask_padding
                elif key.startswith("embedding"):
                    pad_val = self.embedding_padding
                else:
                    pad_val = self.padding_value  

                max_len = max(int(item[key].shape[0]) for item in batch_items)

                collated_batch[f"{key}_lens"] = torch.tensor(
                    [int(item[key].shape[0]) for item in batch_items]
                )
                collated_batch[key] = self._pad_tensor(
                    [item[key] for item in batch_items],
                    target_length=max_len,
                    padding_value=pad_val,
                )
            else:
                values = [item[key] for item in batch_items]
                unique_values = sorted(
                    list(set(values))
                )  
                collated_batch[key] = np.array(unique_values)

                value_to_idx = {v: i for i, v in enumerate(unique_values)}
                collated_batch[f"{key}_keys"] = torch.tensor(
                    [value_to_idx[v] for v in values]
                )
                try:
                    collated_batch[f"{key}_lens"] = np.array([len(v) for v in values])
                except TypeError:
                    pass

        return collated_batch


class ClusterSampler(torch.utils.data.Sampler):
    """Batch sampler that groups samples by cluster.
    
    Groups dataset samples into clusters and samples one item per cluster
    per iteration, useful for stratified or balanced sampling.
    """
    def __init__(self, data, cluster_key, shuffle=False):
        """Initialize ClusterSampler.
        
        Args:
            data: Dataset with cluster labels.
            cluster_key: Key in dataset containing cluster labels.
            shuffle: Whether to shuffle cluster order. Defaults to False.
        """
        super().__init__()

        self.shuffle = shuffle

        df = pd.DataFrame({"cluster": np.array(data[cluster_key])})
        df["idx"] = df.index
        df = df.groupby("cluster").agg({"idx": list}).reset_index()
        self.clusters = pd.Series(df.idx.values, index=df.cluster).to_dict()
        self.cluster_ids = list(self.clusters.keys())

    def __iter__(self):
        """Iterate over sampler indices.
        
        Yields one random sample from each cluster per iteration.
        """
        if self.shuffle:
            np.random.shuffle(self.cluster_ids)

        for cluster_id in self.cluster_ids:
            yield np.random.choice(self.clusters[cluster_id])

    def __len__(self):
        """Return number of clusters (iterations per epoch)."""
        return len(self.cluster_ids)

    def __call__(self):
        """Return self for use as sampler_fn."""
        return self
