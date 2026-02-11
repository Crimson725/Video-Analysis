"""Scheduled cleanup of temporary local job artifacts."""

import logging
import shutil
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
RETAIN_SOURCE_MARKER = ".retain-local-source"


def mark_job_for_source_retention(temp_dir: str, job_id: str) -> None:
    """Mark a job directory so cleanup preserves local source video."""
    job_dir = Path(temp_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    marker = job_dir / RETAIN_SOURCE_MARKER
    marker.write_text("retain-local-source=true\n", encoding="utf-8")


def clear_job_source_retention_marker(temp_dir: str, job_id: str) -> None:
    """Remove a job's source-retention marker."""
    marker = Path(temp_dir) / job_id / RETAIN_SOURCE_MARKER
    marker.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    """Remove file or directory path, best-effort."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _cleanup_retained_job(job_dir: Path) -> None:
    """Cleanup stale job artifacts while preserving retained source video(s)."""
    marker = job_dir / RETAIN_SOURCE_MARKER
    input_dir = job_dir / "input"
    preserved_sources = set(input_dir.glob("source.*")) if input_dir.exists() else set()

    for child in job_dir.iterdir():
        if child == marker or child == input_dir:
            continue
        _remove_path(child)

    if input_dir.exists():
        for child in input_dir.iterdir():
            if child in preserved_sources:
                continue
            _remove_path(child)
        if not any(input_dir.iterdir()):
            input_dir.rmdir()

    if not preserved_sources:
        marker.unlink(missing_ok=True)

    if not any(job_dir.iterdir()):
        job_dir.rmdir()


def cleanup_old_jobs(temp_dir: str, max_age_hours: int = 24) -> None:
    """Delete temporary job directories older than max_age_hours based on mtime."""
    base = Path(temp_dir)
    if not base.exists():
        return
    cutoff = time.time() - (max_age_hours * 3600)
    for item in base.iterdir():
        if item.is_dir():
            mtime = item.stat().st_mtime
            if mtime < cutoff:
                try:
                    marker = item / RETAIN_SOURCE_MARKER
                    if marker.exists():
                        _cleanup_retained_job(item)
                        logger.info(
                            "cleanup.preserve_local_source job_id=%s action=cleanup_stale_artifacts",
                            item.name,
                        )
                    else:
                        shutil.rmtree(item)
                        logger.info("Cleaned up old job directory: %s", item.name)
                except OSError as e:
                    logger.warning("Failed to cleanup %s: %s", item, e)


def setup_scheduler(temp_dir: str) -> BackgroundScheduler:
    """Create and start APScheduler with 24-hour cleanup interval."""
    global _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        cleanup_old_jobs,
        "interval",
        hours=24,
        args=[temp_dir],
        id="cleanup_old_jobs",
    )
    _scheduler.start()
    logger.info("Scheduler started: cleanup every 24 hours")
    return _scheduler


def shutdown_scheduler() -> None:
    """Shut down the scheduler cleanly."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
