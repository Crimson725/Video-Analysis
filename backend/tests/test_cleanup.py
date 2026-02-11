"""Tests for app.cleanup — scheduled cleanup of old job artifacts."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.cleanup import cleanup_old_jobs, setup_scheduler, shutdown_scheduler


class TestCleanupOldJobs:
    def test_old_directory_is_deleted(self, static_dir):
        old_dir = Path(static_dir) / "old-job"
        old_dir.mkdir()
        # Set mtime to 25 hours ago
        old_time = time.time() - (25 * 3600)
        os.utime(old_dir, (old_time, old_time))

        cleanup_old_jobs(static_dir, max_age_hours=24)
        assert not old_dir.exists()

    def test_recent_directory_is_preserved(self, static_dir):
        recent_dir = Path(static_dir) / "recent-job"
        recent_dir.mkdir()
        # mtime is now (< 24h ago)

        cleanup_old_jobs(static_dir, max_age_hours=24)
        assert recent_dir.exists()

    def test_nonexistent_static_dir_handled(self, tmp_path):
        nonexistent = str(tmp_path / "does-not-exist")
        # Should return without error
        cleanup_old_jobs(nonexistent, max_age_hours=24)

    def test_mixed_old_and_new_dirs(self, static_dir):
        old_dir = Path(static_dir) / "old-job"
        old_dir.mkdir()
        old_time = time.time() - (25 * 3600)
        os.utime(old_dir, (old_time, old_time))

        recent_dir = Path(static_dir) / "recent-job"
        recent_dir.mkdir()

        cleanup_old_jobs(static_dir, max_age_hours=24)
        assert not old_dir.exists()
        assert recent_dir.exists()


class TestScheduler:
    @patch("app.cleanup.BackgroundScheduler")
    def test_setup_scheduler_starts(self, mock_scheduler_cls):
        mock_sched = MagicMock()
        mock_scheduler_cls.return_value = mock_sched

        result = setup_scheduler("/tmp/static")

        mock_sched.add_job.assert_called_once()
        mock_sched.start.assert_called_once()
        assert result is mock_sched

    @patch("app.cleanup.BackgroundScheduler")
    def test_shutdown_scheduler_stops(self, mock_scheduler_cls):
        mock_sched = MagicMock()
        mock_scheduler_cls.return_value = mock_sched

        setup_scheduler("/tmp/static")
        shutdown_scheduler()

        mock_sched.shutdown.assert_called_once_with(wait=False)
