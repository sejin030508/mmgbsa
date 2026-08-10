"""Content-addressed, process-safe cache for on-demand protein MSAs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
BuildMSACallback = Callable[[str, Path, Path], None]


def normalize_protein_sequence(sequence: str) -> str:
    """Return the canonical representation used for cache identity."""
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Cannot cache an empty protein sequence")
    return normalized


def sequence_cache_key(sequence: str) -> str:
    normalized = normalize_protein_sequence(sequence)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _first_a3m_sequence(path: Path) -> str | None:
    """Read the first aligned sequence while ignoring A3M insertions."""
    sequence_lines: list[str] = []
    saw_header = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header and sequence_lines:
                    break
                saw_header = True
                continue
            if saw_header:
                sequence_lines.append(line)

    if not sequence_lines:
        return None
    aligned = "".join(sequence_lines)
    return "".join(char for char in aligned if not char.islower()).replace("-", "").upper()


def validate_msa_cache(cache_dir: str | Path, sequence: str) -> tuple[bool, str]:
    """Validate a completed cache entry against the requested sequence."""
    cache_dir = Path(cache_dir)
    normalized = normalize_protein_sequence(sequence)
    expected_hash = sequence_cache_key(normalized)

    manifest_path = cache_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "manifest_missing"
    except (OSError, json.JSONDecodeError):
        return False, "manifest_invalid"

    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False, "schema_mismatch"
    if manifest.get("status") != "complete":
        return False, "not_complete"
    if manifest.get("sequence_sha256") != expected_hash:
        return False, "sequence_hash_mismatch"

    query_path = cache_dir / "query.fasta"
    try:
        query_sequence = _first_a3m_sequence(query_path)
    except OSError:
        return False, "query_unreadable"
    if query_sequence != normalized:
        return False, "query_mismatch"

    for filename in ("non_pairing.a3m", "pairing.a3m"):
        msa_path = cache_dir / "0" / filename
        try:
            if not msa_path.is_file() or msa_path.stat().st_size == 0:
                return False, f"{filename}_missing"
            msa_query = _first_a3m_sequence(msa_path)
        except OSError:
            return False, f"{filename}_unreadable"
        if msa_query != normalized:
            return False, f"{filename}_query_mismatch"

    return True, "ok"


@contextmanager
def _sequence_lock(cache_root: Path, cache_key: str) -> Iterator[None]:
    lock_dir = cache_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_key}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def get_or_create_msa(
    sequence: str,
    cache_root: str | Path,
    build_msa: BuildMSACallback,
    search_mode: str = "protenix",
) -> Path:
    """Return a validated per-sequence MSA directory, building it once if needed."""
    normalized = normalize_protein_sequence(sequence)
    cache_key = sequence_cache_key(normalized)
    cache_root = Path(cache_root).expanduser().resolve() / f"v{CACHE_SCHEMA_VERSION}"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / cache_key

    valid, reason = validate_msa_cache(cache_dir, normalized)
    if valid:
        logger.info("[MSA cache HIT] key=%s length=%d", cache_key[:12], len(normalized))
        return cache_dir / "0"

    logger.info(
        "[MSA cache MISS] key=%s length=%d reason=%s",
        cache_key[:12],
        len(normalized),
        reason,
    )
    with _sequence_lock(cache_root, cache_key):
        valid, _ = validate_msa_cache(cache_dir, normalized)
        if valid:
            logger.info(
                "[MSA cache HIT after wait] key=%s length=%d",
                cache_key[:12],
                len(normalized),
            )
            return cache_dir / "0"

        for stale_temp_dir in cache_root.glob(f".{cache_key}.tmp-*"):
            shutil.rmtree(stale_temp_dir, ignore_errors=True)

        temp_dir = cache_root / f".{cache_key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        query_path = temp_dir / "query.fasta"
        query_path.write_text(f">query\n{normalized}\n", encoding="utf-8")

        try:
            logger.info("[MSA search SUBMIT] key=%s length=%d", cache_key[:12], len(normalized))
            build_msa(normalized, temp_dir, query_path)

            raw_result = temp_dir / "0.a3m"
            if not raw_result.is_file() or raw_result.stat().st_size == 0:
                raise RuntimeError("MSA server did not produce a non-empty 0.a3m result")

            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "complete",
                "sequence_sha256": cache_key,
                "sequence_length": len(normalized),
                "search_mode": search_mode,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            valid, reason = validate_msa_cache(temp_dir, normalized)
            if not valid:
                raise RuntimeError(f"Generated MSA cache failed validation: {reason}")

            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.replace(temp_dir, cache_dir)
            logger.info("[MSA cache WRITE] key=%s path=%s", cache_key[:12], cache_dir)
            return cache_dir / "0"
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
