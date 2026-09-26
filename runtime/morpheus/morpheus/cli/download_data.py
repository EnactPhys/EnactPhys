"""CLI: download the Morpheus video dataset from the Hugging Face Hub.

The dataset is public — no token is required. By default it fetches the real-world
split::

    morpheus-download-data --local-dir ./data

The same repo also holds the resized generated-model videos; select one with
``--allow-patterns`` (or ``--allow-patterns all`` for everything)::

    morpheus-download-data --local-dir ./data_wan \
        --allow-patterns "single_frame_conditioning/WAN-2.1/**" "keyframe_interpolation/WAN-2.1/**"

Setting HF_TOKEN is optional and only helps avoid anonymous-download rate limits::

    export HF_TOKEN=hf_...            # never pass tokens on the command line in shared shells
    morpheus-download-data --local-dir ./data

The dataset is ~36k small files, which can trip Hugging Face's own per-repo rate
limit partway through a download (regardless of token). This is retried
automatically with a backoff (already-downloaded files are kept, so it resumes
rather than starting over) — see --max-retries/--retry-wait-seconds below.
"""

from __future__ import annotations

import argparse

from ..data import (
    DEFAULT_REPO_ID,
    DEFAULT_ALLOW_PATTERNS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_WAIT_SECONDS,
    download_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download the Morpheus real-world video dataset.")
    p.add_argument("--local-dir", required=True, help="Destination directory (downloads resume).")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    p.add_argument("--allow-patterns", nargs="+", default=DEFAULT_ALLOW_PATTERNS,
                   help="Glob patterns to fetch. Use 'all' to download the whole repo.")
    p.add_argument("--revision", default=None, help="Optional git revision / tag.")
    p.add_argument("--token", default=None,
                   help="Optional HF token (dataset is public). Prefer the HF_TOKEN env var over this flag.")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help="Parallel download workers (lower = less likely to trip HF's rate limit).")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help="Retries on HF rate-limit (429) errors before giving up.")
    p.add_argument("--retry-wait-seconds", type=float, default=DEFAULT_RETRY_WAIT_SECONDS,
                   help="Delay between rate-limit retries.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    allow = None if args.allow_patterns == ["all"] else args.allow_patterns
    path = download_dataset(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        allow_patterns=allow,
        token=args.token,
        revision=args.revision,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
