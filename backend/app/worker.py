"""CLI entrypoint for the out-of-process scene AI worker."""

from __future__ import annotations

import logging

from app.config import Settings
from app.scene_ai_worker import SceneAIWorker
from app.scene_task_queue import build_postgres_scene_task_queue


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = Settings.from_env()
    queue = build_postgres_scene_task_queue(dsn=settings.scene_ai_queue_dsn)
    queue.ensure_schema()
    worker = SceneAIWorker.from_settings(settings=settings, queue=queue)
    worker.run_forever()


if __name__ == "__main__":
    main()
