"""
Lightweight dataset parsing utilities for large HuggingFace-style
downloaded dataset bundles (do not scan file contents).

This module provides:
- `parse_readme(dataset_dir)` -> dict with README path and text
- `find_data_files(dataset_dir, patterns)` -> list of candidate data files
- `stream_dataset_from_dir(dataset_dir, patterns)` -> generator yielding examples (uses `datasets` streaming)
- convenience wrappers for the three dataset bundles found in cache/raw

Only parsing/streaming helpers are provided — no eager full-data loads.
"""

from typing import Iterable, List, Dict, Optional
import os
import glob

from datasets import load_dataset


def parse_readme(dataset_dir: str) -> Dict[str, Optional[str]]:
	"""Read README text for a downloaded HuggingFace dataset bundle.

	The function looks for README files at the dataset root and one level
	deep (to avoid scanning huge trees). It does not inspect dataset
	examples or other large files.

	Returns a dict: {'path': path_or_None, 'text': readme_text_or_empty}.
	"""
	candidates = []
	# top-level README names
	for name in ("README.md", "README", "readme.md", "Readme.md"):
		candidates.append(os.path.join(dataset_dir, name))
	# one-level deep README (avoid full recursive walk)
	candidates += glob.glob(os.path.join(dataset_dir, "*/README*"))
	candidates += glob.glob(os.path.join(dataset_dir, "*/readme*"))

	for p in candidates:
		if os.path.exists(p) and os.path.isfile(p):
			try:
				with open(p, "r", encoding="utf-8") as f:
					return {"path": p, "text": f.read()}
			except Exception:
				return {"path": p, "text": None}

	return {"path": None, "text": None}


def find_data_files(dataset_dir: str, patterns: Optional[List[str]] = None) -> List[str]:
	"""Return candidate data file paths without opening them.

	Defaults look for common JSON/JSONL files. Uses glob with recursion
	to locate files but does not read contents.
	"""
	if patterns is None:
		patterns = ["**/*.json", "**/*.jsonl", "*.json", "*.jsonl"]

	found = []
	for pat in patterns:
		glob_pat = os.path.join(dataset_dir, pat)
		found.extend(glob.glob(glob_pat, recursive=True))

	# Deduplicate and sort for deterministic order
	unique = sorted(set(found))
	return unique


def stream_dataset_from_dir(dataset_dir: str, patterns: Optional[List[str]] = None, loader: Optional[object] = None) -> Iterable[Dict]:
	"""Stream examples from a local HuggingFace-style dataset bundle.

	- `dataset_dir`: path to the downloaded dataset folder (cache/raw/...)
	- `patterns`: optional list of glob patterns to locate data files
	- Returns an iterator yielding example dicts lazily.

	This uses `datasets.load_dataset(..., streaming=True)` when available.
	If `datasets` is not installed, raises ImportError.
	"""
	if loader is None:
		loader = load_dataset

	if loader is None:
		raise ImportError("`datasets` library is required for streaming. Install 'datasets'.")

	files = find_data_files(dataset_dir, patterns=patterns)
	if not files:
		raise FileNotFoundError(f"No candidate data files found in {dataset_dir}")

	# Prefer JSON loader for local JSON/JSONL files
	try:
		ds = loader("json", data_files=files, streaming=True)
	except Exception:
		# fallback: let loader try auto-detection
		ds = loader(data_files=files, streaming=True)

	# `ds` may be a DatasetDict or an IterableDataset; normalize to iterator
	if hasattr(ds, "items"):
		# DatasetDict-like -> pick the first split
		for _, split in ds.items():
			for ex in split:
				yield ex
			break
	else:
		for ex in ds:
			yield ex


def stream_belle_multiturn(dataset_dir: str) -> Iterable[Dict]:
	"""Convenience streamer for `BelleGroup_multiturn_chat_0.8M` bundles."""
	patterns = ["**/multiturn_chat_0.8M.json", "**/*.json", "**/*.jsonl"]
	yield from stream_dataset_from_dir(dataset_dir, patterns=patterns)


def stream_belle_school_math(dataset_dir: str) -> Iterable[Dict]:
	"""Convenience streamer for `BelleGroup_school_math_0.25M` bundles."""
	patterns = ["**/school_math_0.25M.json", "**/*.json", "**/*.jsonl"]
	yield from stream_dataset_from_dir(dataset_dir, patterns=patterns)


def stream_bloomvqa(dataset_dir: str) -> Iterable[Dict]:
	"""Convenience streamer for `ygong_BloomVQA` bundles (includes nested `details.json`)."""
	patterns = ["**/details.json", "**/*.json", "**/*.jsonl"]
	yield from stream_dataset_from_dir(dataset_dir, patterns=patterns)


__all__ = [
	"parse_readme",
	"find_data_files",
	"stream_dataset_from_dir",
	"stream_belle_multiturn",
	"stream_belle_school_math",
	"stream_bloomvqa",
]

