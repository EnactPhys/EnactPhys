"""Download the Morpheus video dataset from the Hugging Face Hub.

The dataset repo holds both the real-world reference footage
(``real-world-cropped/<experiment>/<video_dir>/...``) and the resized generated-model
videos (``<conditioning>/<model>/<prompt_type>/<experiment>/<video_dir>/...``). Either
layout is exactly the tree :func:`morpheus.pipeline.process_all_videos` expects. The
default only fetches the real-world split; pass ``allow_patterns`` (e.g.
``["single_frame_conditioning/WAN-2.1/**"]``) to fetch a generated model instead, or
``None`` for everything.

Authentication uses the ``HF_TOKEN`` environment variable (or an explicit ``token``
argument). No token is ever hard-coded.
"""

from __future__ import annotations

import os
import time

# Public dataset repo holding the real-world reference footage (the paper's
# 16 reported real-world experiments). No token required to download.
DEFAULT_REPO_ID = "physics-from-video/morpheus-real-world"
DEFAULT_ALLOW_PATTERNS = ["real-world-cropped*/**"]

# The dataset is ~36k small files (per-frame JPGs). Hugging Face's xet-backed
# file transfer issues one "connection info" request per file, which trips its
# own rate limit (1000 requests / 5 min) well before the general API limit —
# regardless of whether a token is set. The window is ~5 min, so retrying
# sooner than that just re-hits the same limit; the wait below is set to
# roughly match the window so each retry actually lands with fresh quota.
# Downloads resume (already-fetched files are skipped), so retrying reliably
# makes progress — a full first-time download of the whole dataset can still
# take a while (multiple windows), a single/few-experiment download typically
# completes in one attempt.
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 20
DEFAULT_RETRY_WAIT_SECONDS = 300


def _missing_files(repo_id, repo_type, revision, token, allow_patterns, local_dir):
    """Files the repo says should exist (after filtering) that aren't on disk yet.

    Needed because ``snapshot_download`` can hit the rate limit on its own repo
    metadata lookup and, if ``local_dir`` already has *some* content, silently
    return that partial directory as a "success" (a warning, not an exception) —
    so an exception-only retry loop would stop early on incomplete data.
    """
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import filter_repo_objects

    try:
        all_files = list_repo_files(repo_id, repo_type=repo_type, revision=revision, token=token)
    except Exception:
        # Can't even verify right now (e.g. this call itself got rate-limited);
        # be conservative and treat the attempt as incomplete so we retry.
        return ["<unable to verify: repo listing was rate-limited>"]

    wanted = filter_repo_objects(all_files, allow_patterns=allow_patterns)
    return [f for f in wanted if not os.path.exists(os.path.join(local_dir, f))]


def download_dataset(
    local_dir: str,
    repo_id: str = DEFAULT_REPO_ID,
    allow_patterns=DEFAULT_ALLOW_PATTERNS,
    token: str | None = None,
    revision: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
):
    """Snapshot-download (a subset of) a dataset repo to ``local_dir``.

    Parameters
    ----------
    local_dir     : destination directory (created if missing). Downloads resume.
    repo_id       : HF dataset repo id.
    allow_patterns: glob patterns to fetch (default: only the real-world split).
                    Pass ``None`` to fetch everything.
    token         : optional HF token (the dataset is public); falls back to
                    ``$HF_TOKEN`` then cached login, and helps avoid rate limits.
    max_workers   : parallel download workers. Kept modest by default to reduce
                    how hard/fast the xet rate limit above gets hit.
    max_retries   : number of times to retry after a 429 (rate limit) or an
                    incomplete download before giving up and raising. Each retry
                    resumes (already-fetched files are skipped), it doesn't restart.
    retry_wait_seconds: delay between retries.
    """
    from huggingface_hub import snapshot_download

    token = token or os.environ.get("HF_TOKEN")
    os.makedirs(local_dir, exist_ok=True)
    repo_type = "dataset"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                token=token,
                revision=revision,
                max_workers=max_workers,
            )
            missing = _missing_files(repo_id, repo_type, revision, token, allow_patterns, local_dir)
            if not missing:
                return path
            last_error = RuntimeError(
                f"{len(missing)} file(s) still missing after this attempt "
                f"(e.g. {missing[0]}) — snapshot_download likely hit a rate limit "
                "partway through and returned early."
            )
        except Exception as e:
            if "429" not in str(e) and "Too Many Requests" not in str(e):
                raise
            last_error = e

        print(
            f"[morpheus] Download incomplete / rate-limited (attempt {attempt}/{max_retries}); "
            f"retrying in {retry_wait_seconds:.0f}s. Already-downloaded files are kept, "
            "so this resumes rather than starting over."
        )
        time.sleep(retry_wait_seconds)

    raise RuntimeError(
        f"Gave up after {max_retries} retries, still rate-limited / incomplete. "
        "Re-run the same command later — completed files are kept and it will resume."
    ) from last_error
