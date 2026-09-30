"""Per-trial local JSONL writer with verified archive.

Concurrent tau2 workers must not append to one shared file on a CIFS network
mount: that raises OSError errno 16 (EBUSY) and tau2 labels the episode
infrastructure_error. Each trial writes only to its own files under
a local directory, then copies them to the archive root after a checksum
check.

A write error is raised. Callers must not treat a missing stats shard as a
successful ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EBUSY = 16
ENV_LOCAL = "SPECAGENT_TAU2_LOCAL_RUN_DIR"
ENV_ARCHIVE = "SPECAGENT_TAU2_ARCHIVE_DIR"


class TrialIOError(OSError):
    """Ledger or trace write/archive failed. Do not mark the episode successful."""


def _safe_id(value: Any) -> str:
    text = str(value if value is not None else "")
    if not text.strip():
        raise TrialIOError("task_id/trial_id required for per-trial shard path")
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)[:180]


def shard_name(task_id: Any, trial_id: Any) -> str:
    return f"{_safe_id(task_id)}__{_safe_id(trial_id)}.jsonl"


def local_root() -> Path:
    raw = os.environ.get(ENV_LOCAL)
    if raw:
        path = Path(raw)
    else:
        path = Path("/tmp") / "specagent-tau2-io" / str(os.getpid())
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_root(*, stats_path: str | None = None) -> Path | None:
    raw = os.environ.get(ENV_ARCHIVE)
    if raw:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if stats_path:
        path = Path(stats_path).resolve().parent / "shards"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record to a *per-trial* local file; shared files are written by archive_file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), ensure_ascii=False, default=str) + "\n"
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise TrialIOError(f"append failed {path}: {exc}") from exc


def write_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write the episode stats ledger as exactly one JSONL record."""
    line = json.dumps(dict(record), ensure_ascii=False, default=str) + "\n"
    try:
        _atomic_write(path, line.encode("utf-8"))
    except OSError as exc:
        raise TrialIOError(f"stats write failed {path}: {exc}") from exc


def verify_jsonl(path: Path, *, require_n: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrialIOError(f"missing shard {path}")
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise TrialIOError(f"empty shard {path}")
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrialIOError(f"corrupt JSONL {path} line {i}: {exc}") from exc
        if not isinstance(rec, dict):
            raise TrialIOError(f"non-object JSONL {path} line {i}")
        rows.append(rec)
    if require_n is not None and len(rows) != require_n:
        raise TrialIOError(f"{path} has {len(rows)} records, expected {require_n}")
    return rows


def archive_file(src: Path, dest: Path) -> str:
    rows = verify_jsonl(src)
    data = src.read_bytes()
    digest = _sha256_bytes(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_exc: OSError | None = None
    for attempt in range(3):
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            last_exc = None
            break
        except OSError as exc:
            last_exc = exc
            if getattr(exc, "errno", None) != EBUSY or attempt == 2:
                raise TrialIOError(f"archive copy failed {src} -> {dest}: {exc}") from exc
            time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise TrialIOError(f"archive copy failed {src} -> {dest}: {last_exc}") from last_exc
    copied = dest.read_bytes()
    if _sha256_bytes(copied) != digest:
        raise TrialIOError(f"archive checksum mismatch {dest}")
    verify_jsonl(dest, require_n=len(rows))
    return digest


class TrialShardWriter:
    def __init__(self, *, stats_env_path: str | None) -> None:
        self.stats_env_path = stats_env_path
        self.local = local_root()
        self.archive = archive_root(stats_path=stats_env_path)

    def enabled(self) -> bool:
        return bool(self.stats_env_path)

    def _ids(self, task_id: Any, trial_id: Any) -> tuple[Any, Any]:
        if task_id is None or trial_id in (None, ""):
            raise TrialIOError("refusing to write a ledger/trace without task_id and trial_id")
        return task_id, trial_id

    def local_path(self, kind: str, task_id: Any, trial_id: Any) -> Path:
        task_id, trial_id = self._ids(task_id, trial_id)
        return self.local / kind / shard_name(task_id, trial_id)

    def archive_path(self, kind: str, task_id: Any, trial_id: Any) -> Path:
        if self.archive is None:
            raise TrialIOError("archive root is not configured")
        task_id, trial_id = self._ids(task_id, trial_id)
        return self.archive / kind / shard_name(task_id, trial_id)

    def append_trace(self, record: Mapping[str, Any], *, task_id: Any, trial_id: Any) -> Path:
        path = self.local_path("verdicts", task_id, trial_id)
        rec = dict(record)
        rec.setdefault("task_id", task_id)
        rec.setdefault("trial_id", trial_id)
        append_jsonl(path, rec)
        return path

    def write_stats(self, record: Mapping[str, Any], *, task_id: Any, trial_id: Any) -> Path:
        path = self.local_path("stats", task_id, trial_id)
        rec = dict(record)
        rec.setdefault("task_id", task_id)
        rec.setdefault("trial_id", trial_id)
        write_jsonl_record(path, rec)
        return path

    def archive_trial(self, *, task_id: Any, trial_id: Any) -> dict[str, str]:
        """Copy local stats (required) and verdicts (if present) to the archive.

        Missing stats is a hard error: the ledger must not be silently dropped.
        """
        stats_src = self.local_path("stats", task_id, trial_id)
        if not stats_src.is_file():
            raise TrialIOError(
                f"missing stats ledger for task={task_id} trial={trial_id} at {stats_src}"
            )
        verify_jsonl(stats_src, require_n=1)
        digests = {
            "stats": archive_file(stats_src, self.archive_path("stats", task_id, trial_id)),
        }
        verdicts_src = self.local_path("verdicts", task_id, trial_id)
        if verdicts_src.is_file():
            digests["verdicts"] = archive_file(
                verdicts_src, self.archive_path("verdicts", task_id, trial_id)
            )
        return digests


def collect_kind(kind: str, *, stats_path: str | None = None) -> list[dict[str, Any]]:
    """Load per-trial shards. Prefer archive after a successful archive_trial()."""
    writer = configure_from_stats_env(
        stats_path
        or os.environ.get("SPECAGENT_STATS")
        or os.environ.get("SPECAGENT_BASELINE_STATS")
    )
    if writer is None:
        raise TrialIOError("no stats path configured; cannot collect shards")
    archive_dir = writer.archive / kind if writer.archive is not None else None
    local_dir = writer.local / kind
    chosen: Path | None = None
    if archive_dir is not None and archive_dir.is_dir() and any(archive_dir.glob("*.jsonl")):
        chosen = archive_dir
    elif local_dir.is_dir() and any(local_dir.glob("*.jsonl")):
        chosen = local_dir
    if chosen is None:
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(chosen.glob("*.jsonl")):
        rows.extend(verify_jsonl(path))
    return rows


def merge_shards(kind_dir: Path, dest: Path) -> int:
    """Single-writer concat of per-trial shards. Do not call from conc=N workers."""
    files = sorted(kind_dir.glob("*.jsonl"))
    n = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for path in files:
            rows = verify_jsonl(path)
            for rec in rows:
                out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
    os.replace(tmp, dest)
    return n


def configure_from_stats_env(stats_path: str | None) -> TrialShardWriter | None:
    if not stats_path:
        return None
    if not os.environ.get(ENV_LOCAL):
        os.environ[ENV_LOCAL] = str(Path("/tmp") / "specagent-tau2-io" / str(os.getpid()))
    if not os.environ.get(ENV_ARCHIVE):
        os.environ[ENV_ARCHIVE] = str(Path(stats_path).resolve().parent / "shards")
    return TrialShardWriter(stats_env_path=stats_path)
