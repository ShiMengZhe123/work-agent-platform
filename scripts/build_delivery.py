"""Build a clean source/demo ZIP without virtualenv, caches, or secrets."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "WorkMateAI_delivery.zip"
INCLUDE_DIRS = [
    "workmate",
    "scripts",
    "tests",
    ".streamlit",
    "data",
    "artifacts",
    "outputs",
    "screenshots",
    "logs",
]
INCLUDE_FILES = [
    "app.py",
    "README.md",
    "DELIVERY.md",
    "requirements.txt",
    "requirements-lock.txt",
    "pytest.ini",
    "run.ps1",
    ".env.example",
    ".gitignore",
    "test_results.log",
    "demo_smoke.log",
    "health_check.log",
    "model_info.txt",
]
EXCLUDED_PARTS = {
    ".venv",
    ".cache",
    ".tmp",
    ".pytest_cache",
    ".playwright-cli",
    "__pycache__",
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name == ".env.local" or path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def main() -> None:
    with zipfile.ZipFile(
        OUTPUT,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for file_name in INCLUDE_FILES:
            path = ROOT / file_name
            if path.exists() and should_include(path):
                archive.write(path, path.relative_to(ROOT))
        for directory_name in INCLUDE_DIRS:
            directory = ROOT / directory_name
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if should_include(path):
                    archive.write(path, path.relative_to(ROOT))

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"DELIVERY={OUTPUT}")
    print(f"BYTES={OUTPUT.stat().st_size}")
    print(f"SHA256={digest}")
    with zipfile.ZipFile(OUTPUT) as archive:
        print(f"FILES={len(archive.namelist())}")
        bad = archive.testzip()
        print(f"ZIP_TEST={'OK' if bad is None else bad}")


if __name__ == "__main__":
    main()
