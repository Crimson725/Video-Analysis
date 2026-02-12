"""Shared fixtures for integration tests using real models and real video."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from app.config import Settings
from app.models import ModelLoader
from app.scene import detect_scenes, extract_keyframes
from app.storage import MediaStoreError, R2MediaStore


# ---------------------------------------------------------------------------
# Resolve the test video path relative to the repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_VIDEO = _REPO_ROOT / "Test Videos" / "WhatCarCanYouGetForAGrand.mp4"
_R2_ENV_FILE = _REPO_ROOT / "backend" / ".env.r2"
_CORPUS_COMPOSE_FILE = _REPO_ROOT / "backend" / "docker-compose.corpus.yml"
_CORPUS_COMPOSE_PROJECT = "video-analysis-corpus-itest"
_DEFAULT_CORPUS_TEST_NEO4J_BOLT_PORT = 17687
_DEFAULT_CORPUS_TEST_PGVECTOR_PORT = 15433
_OUTPUT_CONTENT_EXPECTATIONS = (
    Path(__file__).resolve().parent / "fixtures" / "output_content_expectations.json"
)


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE env vars from a dotenv-style file."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _read_env(name: str, default: str = "") -> str:
    """Return a trimmed environment variable value."""
    return os.getenv(name, default).strip()


def _skip_with_command(message: str) -> None:
    """Skip with an actionable command for enabling external-api integration tests."""
    command = (
        "cd backend && "
        "GOOGLE_API_KEY='<your-gemini-key>' ENABLE_SCENE_UNDERSTANDING_PIPELINE=true "
        "uv run pytest tests/integration/test_output_content_e2e_integration.py "
        "-m 'integration and external_api' -vv"
    )
    pytest.skip(f"{message}. Enable with: {command}")


def _skip_corpus_e2e_with_command(message: str) -> None:
    """Skip with an actionable command for enabling no-LLM corpus e2e tests."""
    command = (
        "cd backend && "
        "RUN_CORPUS_E2E_INTEGRATION=1 ENABLE_SCENE_UNDERSTANDING_PIPELINE=false "
        "ENABLE_CORPUS_PIPELINE=true ENABLE_CORPUS_INGEST=true "
        "uv run pytest tests/integration/test_no_llm_corpus_e2e_integration.py -m integration -vv"
    )
    pytest.skip(f"{message}. Enable with: {command}")


def _truncate_message(text: str, *, limit: int = 220) -> str:
    """Return a single-line message trimmed for readable pytest skip output."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _run_command(
    command: list[str],
    *,
    timeout_seconds: int = 120,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Run a shell command and return (ok, detail)."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
    except FileNotFoundError:
        return False, f"command not found: {command[0]}"
    except Exception as exc:
        return False, f"{command[0]} failed to execute ({exc})"

    output = result.stderr.strip() or result.stdout.strip()
    if result.returncode == 0:
        return True, _truncate_message(output or "ok")
    return False, _truncate_message(output or f"exit code {result.returncode}")


def _is_loopback_host(host: str | None) -> bool:
    """Return whether a hostname points to local loopback."""
    return host in {"127.0.0.1", "localhost", "::1"}


def _corpus_runtime_targets_local_backends(corpus_e2e_runtime: dict[str, str]) -> bool:
    """Return True when runtime config points to local Neo4j + pgvector endpoints."""
    neo4j_host = urlparse(corpus_e2e_runtime["neo4j_uri"]).hostname
    pg_host = urlparse(corpus_e2e_runtime["pgvector_dsn"]).hostname
    return _is_loopback_host(neo4j_host) and _is_loopback_host(pg_host)


def _derive_neo4j_http_port(bolt_port: int) -> int:
    """Return a deterministic HTTP port paired with a Neo4j Bolt host port."""
    if bolt_port > 213:
        return bolt_port - 213
    return 17474


def _compose_env_for_corpus_runtime(corpus_e2e_runtime: dict[str, str]) -> dict[str, str]:
    """Build compose environment overrides from runtime connection settings."""
    neo4j_uri = urlparse(corpus_e2e_runtime["neo4j_uri"])
    pg_uri = urlparse(corpus_e2e_runtime["pgvector_dsn"])

    neo4j_bolt_port = neo4j_uri.port or _DEFAULT_CORPUS_TEST_NEO4J_BOLT_PORT
    pg_port = pg_uri.port or _DEFAULT_CORPUS_TEST_PGVECTOR_PORT

    neo4j_username = corpus_e2e_runtime["neo4j_username"] or "neo4j"
    neo4j_password = corpus_e2e_runtime["neo4j_password"] or "local-dev-password"

    pg_user = pg_uri.username or "video_analysis"
    pg_password = pg_uri.password or "video_analysis"
    pg_database = pg_uri.path.lstrip("/") or "video_analysis"

    compose_env = os.environ.copy()
    compose_env.update(
        {
            "CORPUS_NEO4J_AUTH": f"{neo4j_username}/{neo4j_password}",
            "CORPUS_NEO4J_BOLT_PORT": str(neo4j_bolt_port),
            "CORPUS_NEO4J_HTTP_PORT": str(_derive_neo4j_http_port(neo4j_bolt_port)),
            "CORPUS_PGVECTOR_PORT": str(pg_port),
            "CORPUS_PGVECTOR_USER": pg_user,
            "CORPUS_PGVECTOR_PASSWORD": pg_password,
            "CORPUS_PGVECTOR_DB": pg_database,
        }
    )
    return compose_env


def _try_start_podman_machine() -> None:
    """Best-effort Podman machine startup for macOS/local development."""
    if not shutil.which("podman"):
        return
    _run_command(["podman", "machine", "start", "podman-machine-default"], timeout_seconds=90)


def _start_local_corpus_stack(corpus_e2e_runtime: dict[str, str]) -> tuple[bool, str]:
    """Try to start local Neo4j + pgvector stack using Podman first, Docker fallback."""
    if not _corpus_runtime_targets_local_backends(corpus_e2e_runtime):
        return False, "auto-start skipped because configured DB endpoints are non-local"
    if not _CORPUS_COMPOSE_FILE.is_file():
        return False, f"compose file not found: {_CORPUS_COMPOSE_FILE}"
    compose_env = _compose_env_for_corpus_runtime(corpus_e2e_runtime)

    runners: list[tuple[str, list[str]]] = []
    if shutil.which("podman"):
        _try_start_podman_machine()
        runners.append(
            (
                "podman",
                [
                    "podman",
                    "compose",
                    "-p",
                    _CORPUS_COMPOSE_PROJECT,
                    "-f",
                    str(_CORPUS_COMPOSE_FILE),
                    "up",
                    "-d",
                ],
            )
        )
    if shutil.which("docker"):
        runners.append(
            (
                "docker",
                [
                    "docker",
                    "compose",
                    "-p",
                    _CORPUS_COMPOSE_PROJECT,
                    "-f",
                    str(_CORPUS_COMPOSE_FILE),
                    "up",
                    "-d",
                ],
            )
        )
    if not runners:
        return False, "neither podman nor docker is available"

    failures: list[str] = []
    for runner_name, command in runners:
        # Ensure this test stack starts from a clean DB volume state.
        down_cmd = command[:-2] + ["down", "-v", "--remove-orphans"]
        _run_command(down_cmd, timeout_seconds=120, env=compose_env)

        ok, detail = _run_command(command, timeout_seconds=180, env=compose_env)
        if ok:
            return True, f"{runner_name} compose up -d succeeded"
        failures.append(f"{runner_name}: {detail}")
    return False, "; ".join(failures)


def _probe_neo4j(corpus_e2e_runtime: dict[str, str]) -> str | None:
    """Return probe error for Neo4j, or None when reachable."""
    from neo4j import GraphDatabase

    driver = None
    try:
        driver = GraphDatabase.driver(
            corpus_e2e_runtime["neo4j_uri"],
            auth=(
                corpus_e2e_runtime["neo4j_username"],
                corpus_e2e_runtime["neo4j_password"],
            ),
        )
        with driver.session(database=corpus_e2e_runtime["neo4j_database"]) as session:
            session.run("RETURN 1").single()
        return None
    except Exception as exc:
        return f"Neo4j is not reachable ({exc})"
    finally:
        if driver is not None:
            driver.close()


def _probe_pgvector(corpus_e2e_runtime: dict[str, str]) -> str | None:
    """Return probe error for PostgreSQL+pgvector, or None when reachable."""
    import psycopg

    try:
        with psycopg.connect(corpus_e2e_runtime["pgvector_dsn"]) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return None
    except Exception as exc:
        return f"PostgreSQL/pgvector is not reachable ({exc})"


def _probe_corpus_backends(corpus_e2e_runtime: dict[str, str]) -> tuple[bool, str]:
    """Return (ready, detail) for local Neo4j and pgvector probe checks."""
    errors = [
        error
        for error in (
            _probe_neo4j(corpus_e2e_runtime),
            _probe_pgvector(corpus_e2e_runtime),
        )
        if error
    ]
    if errors:
        return False, "; ".join(errors)
    return True, "ready"


def _is_non_retryable_probe_error(detail: str) -> bool:
    """Return True when probe detail indicates a startup retry will not help."""
    lowered = detail.lower()
    if "password authentication failed" in lowered:
        return True
    if "authentication failed" in lowered and "neo4j" in lowered:
        return True
    if "role \"" in lowered and "does not exist" in lowered:
        return True
    if "database \"" in lowered and "does not exist" in lowered:
        return True
    return False


def _wait_for_corpus_backends(corpus_e2e_runtime: dict[str, str], *, timeout_seconds: int = 90) -> tuple[bool, str]:
    """Wait until Neo4j and pgvector become ready, or timeout with last probe error."""
    deadline = time.time() + timeout_seconds
    last_error = "backends still starting"
    while time.time() < deadline:
        ready, detail = _probe_corpus_backends(corpus_e2e_runtime)
        if ready:
            return True, detail
        last_error = detail
        if _is_non_retryable_probe_error(detail):
            return False, detail
        time.sleep(2)
    return False, last_error


# ---------------------------------------------------------------------------
# 2.1 — Test video path fixture (session-scoped, skips if missing)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_video_path():
    """Return the absolute path to the test video, or skip if not present."""
    if not _TEST_VIDEO.is_file():
        pytest.skip(f"Test video not found: {_TEST_VIDEO}")
    return str(_TEST_VIDEO)


@pytest.fixture(scope="session")
def corpus_e2e_real_video_path(test_video_path: str) -> str:
    """Return canonical real video path for no-LLM corpus e2e tests."""
    canonical = _TEST_VIDEO.resolve()
    actual = Path(test_video_path).resolve()
    if actual != canonical:
        pytest.fail(
            "No-LLM corpus e2e tests require canonical real video "
            f"at {canonical}, got {actual}"
        )
    if actual.stat().st_size <= 0:
        pytest.fail(f"Canonical test video is empty: {actual}")
    return str(actual)


# ---------------------------------------------------------------------------
# 2.2 — Real model loader fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def models():
    """Load real YOLO and MTCNN models once per session."""
    return ModelLoader.get()


# ---------------------------------------------------------------------------
# 2.3 — Scene boundaries fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def scenes(test_video_path):
    """Detect scenes from the real test video once per session."""
    return detect_scenes(test_video_path)


# ---------------------------------------------------------------------------
# 2.4 — Keyframes fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def keyframes(test_video_path, scenes):
    """Extract keyframes from the real test video once per session."""
    frames = extract_keyframes(test_video_path, scenes)
    assert len(frames) > 0, "No keyframes extracted from test video"
    return frames


# ---------------------------------------------------------------------------
# 2.5 — Single sample frame fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_frame(keyframes):
    """Return the first extracted keyframe for use in analysis tests."""
    return keyframes[0]


# ---------------------------------------------------------------------------
# 2.6 — Temporary static directory fixture (function-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture()
def static_dir(tmp_path):
    """Return a fresh temporary directory usable as `static_dir`."""
    return str(tmp_path)


@pytest.fixture(scope="session")
def r2_store():
    """Return a real R2 media store; missing credentials are a hard failure."""
    _load_env_file(_R2_ENV_FILE)
    settings = Settings.from_env()
    missing = settings.missing_r2_fields()
    if missing:
        raise RuntimeError(f"Missing R2 settings for integration tests: {', '.join(missing)}")

    return R2MediaStore(
        account_id=settings.r2_account_id,
        bucket=settings.r2_bucket,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        default_url_ttl_seconds=settings.r2_url_ttl_seconds,
    )


@pytest.fixture()
def r2_cleanup_keys(r2_store):
    """Collect created R2 keys and delete them after the test."""
    created_keys: list[str] = []
    yield created_keys

    for object_key in created_keys:
        try:
            r2_store.delete_object(object_key)
        except MediaStoreError:
            # Best effort cleanup; failures are reported by read assertions in tests.
            pass


@pytest.fixture()
def r2_test_job_id():
    """Return a unique job id for R2 integration artifact tests."""
    return f"r2-itest-{uuid4().hex}"


@pytest.fixture(scope="session")
def corpus_e2e_runtime():
    """Return runtime config for no-LLM corpus e2e tests, or skip with guidance."""
    if _read_env("RUN_CORPUS_E2E_INTEGRATION") != "1":
        _skip_corpus_e2e_with_command("RUN_CORPUS_E2E_INTEGRATION=1 is required")

    try:
        import neo4j  # noqa: F401
        import psycopg  # noqa: F401
    except Exception as exc:
        _skip_corpus_e2e_with_command(
            f"Missing DB client dependency for corpus e2e integration tests ({exc})"
        )

    neo4j_uri = (
        _read_env("NEO4J_URI", f"bolt://127.0.0.1:{_DEFAULT_CORPUS_TEST_NEO4J_BOLT_PORT}")
        or f"bolt://127.0.0.1:{_DEFAULT_CORPUS_TEST_NEO4J_BOLT_PORT}"
    )
    neo4j_username = _read_env("NEO4J_USERNAME", "neo4j") or "neo4j"
    neo4j_password = _read_env("NEO4J_PASSWORD", "local-dev-password") or "local-dev-password"
    neo4j_database = _read_env("NEO4J_DATABASE", "neo4j") or "neo4j"
    pgvector_dsn = (
        _read_env(
            "PGVECTOR_DSN",
            f"postgresql://video_analysis:video_analysis@127.0.0.1:{_DEFAULT_CORPUS_TEST_PGVECTOR_PORT}/video_analysis",
        )
        or f"postgresql://video_analysis:video_analysis@127.0.0.1:{_DEFAULT_CORPUS_TEST_PGVECTOR_PORT}/video_analysis"
    )

    return {
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_database": neo4j_database,
        "pgvector_dsn": pgvector_dsn,
    }


@pytest.fixture(scope="session")
def corpus_e2e_backend_probe(corpus_e2e_runtime):
    """Verify local Neo4j and pgvector are reachable for no-LLM corpus e2e tests."""
    ready, detail = _probe_corpus_backends(corpus_e2e_runtime)
    if not ready:
        started, startup_detail = _start_local_corpus_stack(corpus_e2e_runtime)
        if started:
            ready, detail = _wait_for_corpus_backends(corpus_e2e_runtime)
        else:
            detail = f"{detail}; auto-start failed ({startup_detail})"
    if not ready:
        _skip_corpus_e2e_with_command(
            "Neo4j/pgvector is not reachable after auto-start attempt "
            f"({_truncate_message(detail, limit=500)})"
        )

    return corpus_e2e_runtime


@pytest.fixture()
def corpus_e2e_job_id():
    """Return a deterministic namespace prefix for no-LLM corpus e2e test runs."""
    return f"corpus-e2e-{uuid4().hex[:12]}"


@pytest.fixture()
def corpus_e2e_db_cleanup(corpus_e2e_backend_probe):
    """Track and clean up DB rows written by a no-LLM corpus e2e test."""
    from neo4j import GraphDatabase
    import psycopg

    tracked_node_ids: set[str] = set()
    tracked_claim_ids: set[str] = set()
    tracked_chunk_ids: set[str] = set()

    def _track(*, node_ids: list[str], claim_ids: list[str], chunk_ids: list[str]) -> None:
        tracked_node_ids.update(node_ids)
        tracked_claim_ids.update(claim_ids)
        tracked_chunk_ids.update(chunk_ids)

    yield _track

    if tracked_chunk_ids:
        chunk_ids = sorted(tracked_chunk_ids)
        try:
            with psycopg.connect(corpus_e2e_backend_probe["pgvector_dsn"]) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM corpus_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
                conn.commit()
        except Exception:
            pass

    if tracked_node_ids or tracked_claim_ids:
        try:
            driver = GraphDatabase.driver(
                corpus_e2e_backend_probe["neo4j_uri"],
                auth=(
                    corpus_e2e_backend_probe["neo4j_username"],
                    corpus_e2e_backend_probe["neo4j_password"],
                ),
            )
            with driver.session(database=corpus_e2e_backend_probe["neo4j_database"]) as session:
                if tracked_node_ids:
                    session.run(
                        "MATCH (n:CorpusNode) WHERE n.node_id IN $node_ids DETACH DELETE n",
                        node_ids=sorted(tracked_node_ids),
                    )
                if tracked_claim_ids:
                    session.run(
                        "MATCH (c:CorpusClaim) WHERE c.claim_id IN $claim_ids DELETE c",
                        claim_ids=sorted(tracked_claim_ids),
                    )
            driver.close()
        except Exception:
            pass


@pytest.fixture(scope="session")
def gemini_api_key():
    """Return Gemini API key for external-API integration tests, or skip."""
    key = _read_env("GOOGLE_API_KEY")
    if not key:
        _skip_with_command(
            "GOOGLE_API_KEY is required for Gemini-backed output-content integration tests"
        )
    return key


@pytest.fixture(scope="session")
def gemini_probe(gemini_api_key):
    """Validate Gemini client can be constructed for external-API tests."""
    from app.video_understanding import GeminiSceneLLMClient

    scene_model_id = _read_env("SCENE_MODEL_ID", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite"
    synopsis_model_id = (
        _read_env("SYNOPSIS_MODEL_ID", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite"
    )
    try:
        GeminiSceneLLMClient(
            google_api_key=gemini_api_key,
            scene_model_id=scene_model_id,
            synopsis_model_id=synopsis_model_id,
        )
    except Exception as exc:
        pytest.skip(f"Skipping Gemini-backed integration tests: client setup failed ({exc}).")
    return {
        "scene_model_id": scene_model_id,
        "synopsis_model_id": synopsis_model_id,
    }


@pytest.fixture()
def synopsis_e2e_settings(monkeypatch, gemini_api_key):
    """Return settings for synopsis E2E tests with scene understanding enabled."""
    _load_env_file(_R2_ENV_FILE)
    monkeypatch.setenv("GOOGLE_API_KEY", gemini_api_key)
    monkeypatch.setenv("ENABLE_SCENE_UNDERSTANDING_PIPELINE", "true")
    settings = Settings.from_env()

    missing_r2 = settings.missing_r2_fields()
    if missing_r2:
        _skip_with_command(
            "Missing R2 settings for synopsis integration tests: " + ", ".join(missing_r2)
        )
    missing_llm = settings.missing_llm_fields()
    if missing_llm:
        _skip_with_command(
            "Missing LLM settings for synopsis integration tests: " + ", ".join(missing_llm)
        )
    return settings


@pytest.fixture(scope="session")
def canonical_output_content_expectations():
    """Load canonical output-content rubric thresholds and topic anchors."""
    if not _OUTPUT_CONTENT_EXPECTATIONS.is_file():
        raise RuntimeError(f"Missing expectations file: {_OUTPUT_CONTENT_EXPECTATIONS}")

    raw = json.loads(_OUTPUT_CONTENT_EXPECTATIONS.read_text(encoding="utf-8"))
    anchors = [anchor.lower().strip() for anchor in raw.get("topic_anchors", []) if anchor.strip()]
    if not anchors:
        raise RuntimeError("output_content_expectations.json must define non-empty topic_anchors")

    return {
        "video_filename": str(raw.get("video_filename", "WhatCarCanYouGetForAGrand.mp4")).strip(),
        "min_scene_narrative_chars": int(raw.get("min_scene_narrative_chars", 40)),
        "min_synopsis_chars": int(raw.get("min_synopsis_chars", 80)),
        "min_key_moment_chars": int(raw.get("min_key_moment_chars", 6)),
        "min_topic_anchor_matches": int(raw.get("min_topic_anchor_matches", 1)),
        "topic_anchors": anchors,
    }
