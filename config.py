"""Environment-backed configuration for the local-first DiscreteMemory product."""

import os
from pathlib import Path, PureWindowsPath

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def _local_path(value):
    """Accept Windows drive paths when this checkout runs under WSL."""
    raw = str(value).strip()
    windows_path = PureWindowsPath(raw)
    if os.name != "nt" and windows_path.is_absolute():
        drive = windows_path.drive
        if len(drive) == 2 and drive[1] == ":":
            mount = Path("/mnt") / drive[0].lower()
            if mount.is_dir():
                return mount.joinpath(*windows_path.parts[1:])
    return Path(raw).expanduser()


def _resolve_project_path(value, project_root=None):
    """Keep data/state paths stable across IDE, CLI and WSL working dirs."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("project path must not be empty")
    path = _local_path(raw)
    if not path.is_absolute():
        path = Path(project_root or PROJECT_ROOT) / path
    return str(path.resolve(strict=False))


def _resolve_model_reference(value, project_root=None):
    """Resolve local model paths without rewriting Hugging Face model IDs.

    PyCharm, ``uvicorn`` and WSL shells do not necessarily share the same
    current working directory.  Explicit/available local paths are therefore
    anchored to the repository, while references such as
    ``BAAI/bge-small-zh-v1.5`` remain valid remote model identifiers.
    """

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("model reference must not be empty")
    path = _local_path(raw)
    if path.is_absolute():
        return str(path.resolve(strict=False))
    root = Path(project_root or PROJECT_ROOT)
    candidate = root / path
    if raw.startswith((".", "~")) or candidate.exists():
        return str(candidate.resolve(strict=False))
    return raw


def _load_project_environment():
    """Load the project .env and an optional explicitly configured env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    external = os.getenv("DISCRETE_MEMORY_ENV_FILE", "").strip()
    if external:
        path = Path(_resolve_project_path(external))
        if not path.is_file():
            raise FileNotFoundError(f"external environment file does not exist: {path}")
        # Explicit process environment values retain their normal precedence.
        load_dotenv(path)


_load_project_environment()


class Config:
    EMBED_MODEL_PATH = _resolve_model_reference(
        os.getenv("EMBED_MODEL_PATH", "./models/bge-small-zh-v1.5")
    )
    OLLAMA_URL = os.getenv(
        "OLLAMA_URL", "http://localhost:11434/api/generate"
    )
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

    SHORT_TERM_LIMIT = int(os.getenv("SHORT_TERM_LIMIT", "5"))

    DEFAULT_WORKSPACE_ID = os.getenv("DEFAULT_WORKSPACE_ID", "local")
    PRODUCT_DB_PATH = _resolve_project_path(
        os.getenv("PRODUCT_DB_PATH", "./storage/discrete_memory.db")
    )
    DOCUMENTS_PATH = _resolve_project_path(
        os.getenv("DOCUMENTS_PATH", "./storage/documents")
    )
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    BASE_DB_PATH = _resolve_project_path(
        os.getenv("BASE_DB_PATH", "./storage/career_knowledge")
    )
    MEMORY_DB_PATH = _resolve_project_path(
        os.getenv("MEMORY_DB_PATH", "./storage/user_memory")
    )
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    OVERLAP_SENTENCE = int(os.getenv("OVERLAP_SENTENCE", "1"))

    BASE_MAX_DISTANCE = float(os.getenv("BASE_MAX_DISTANCE", "1.2"))
    MEMORY_MAX_DISTANCE = float(os.getenv("MEMORY_MAX_DISTANCE", "1.5"))
    BASE_RECALL_K = int(os.getenv("BASE_RECALL_K", "3"))
    MEMORY_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "3"))
