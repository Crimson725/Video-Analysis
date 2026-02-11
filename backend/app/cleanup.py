"""Scheduled cleanup of temporary local job artifacts."""

import logging
import shutil
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


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
