
"""Download recommended datasets from HuggingFace.

Usage examples:
  python data/download.py --all
  python data/download.py --datasets belle_math bloomvqa --outdir data/raw

Dependencies:
  pip install huggingface_hub

This script uses `huggingface_hub.snapshot_download` to fetch dataset repo contents
and copies them to `--outdir`. If `wget` is available and a direct URL is provided,
the script will attempt to use it as a fallback.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

try:
	from huggingface_hub import snapshot_download
except Exception as e:  # pragma: no cover - import error handled at runtime
	snapshot_download = None  # type: ignore

LOG = logging.getLogger(__name__)


DATASETS: Dict[str, str] = {
	"belle_math": "BelleGroup/school_math_0.25M",
	"belle_chat": "BelleGroup/multiturn_chat_0.8M",
	"bloomvqa": "ygong/BloomVQA",
}


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def download_hf_snapshot(repo_id: str, out_dir: Path) -> Path:
	"""Download a HuggingFace dataset repo using snapshot_download and copy it to out_dir."""
	if snapshot_download is None:
		raise RuntimeError("huggingface_hub is required: pip install huggingface_hub")

	LOG.info("Fetching %s via HuggingFace snapshot_download...", repo_id)
	cache_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

	dest = out_dir / repo_id.replace("/", "_")
	if dest.exists():
		LOG.info("Destination %s already exists, skipping copy.", dest)
		return dest

	LOG.info("Copying files from cache %s to %s", cache_path, dest)
	shutil.copytree(cache_path, dest)
	return dest


def download_url(url: str, out_dir: Path) -> Path:
	"""Download a direct URL using wget as a fallback. Returns path to downloaded file."""
	ensure_dir(out_dir)
	LOG.info("Downloading URL %s to %s", url, out_dir)
	try:
		subprocess.run(["wget", "-c", "-P", str(out_dir), url], check=True)
	except FileNotFoundError:
		raise RuntimeError("wget not found; please install wget or use huggingface_hub instead")
	return out_dir


def download_selected(names: list[str], outdir: str) -> None:
	out = Path(outdir)
	ensure_dir(out)

	for name in names:
		if name not in DATASETS:
			LOG.warning("Unknown dataset key: %s (skipping)", name)
			continue

		repo = DATASETS[name]
		try:
			download_hf_snapshot(repo, out)
			LOG.info("Downloaded %s -> %s", repo, out)
		except Exception as e:
			LOG.warning("snapshot_download failed for %s: %s", repo, e)
			LOG.warning("No further automatic fallback available for dataset repos.")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Download datasets referenced in docs/plan.md")
	p.add_argument("--outdir", default="data/raw", help="Output directory")
	p.add_argument("--datasets", nargs="*", help="Dataset keys to download (or --all)")
	p.add_argument("--all", action="store_true", help="Download all recommended datasets")
	p.add_argument("--verbose", action="store_true")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

	if args.all or not args.datasets:
		to_download = list(DATASETS.keys())
	else:
		to_download = args.datasets

	LOG.info("BelleGroup org page: https://huggingface.co/BelleGroup")
	download_selected(to_download, args.outdir)
	LOG.info("Done. Check %s for downloaded dataset folders.", args.outdir)


if __name__ == "__main__":
	main()

