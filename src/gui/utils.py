from __future__ import annotations

import functools
import os
import re
import shlex
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".log",
    ".py",
    ".ini",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def norm_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def list_files(base_dirs: Iterable[Path], recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        globber = base.rglob("*") if recursive else base.glob("*")
        for p in globber:
            if p.is_file():
                files.append(p)
    files.sort(key=lambda p: norm_path(p).lower())
    return files


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def read_text_preview(path: Path, max_lines: int = 120, max_chars: int = 25000) -> str:
    if not path.exists() or not path.is_file():
        return "File not found."

    chunks: list[str] = []
    total = 0
    line_count = 0
    truncated = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                chunks.append(line)
                total += len(line)
                line_count += 1
                if line_count >= max_lines or total >= max_chars:
                    truncated = True
                    break
    except OSError as exc:
        return f"Failed to read file: {exc}"

    text = "".join(chunks)
    if truncated:
        text += "\n\n[Preview truncated]"
    return text


def discover_dataset_files() -> list[Path]:
    return list_files([repo_root() / "data"], recursive=False)


def discover_readable_outputs() -> list[Path]:
    root = repo_root()
    return list_files([root / "plots", root / "results", root / "result"], recursive=True)


@functools.lru_cache(maxsize=1)
def discover_cli_modules() -> list[str]:
    modules: list[str] = []
    root = repo_root() / "src"
    if not root.exists():
        return modules

    for file in root.rglob("*.py"):
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "if __name__" in content and "__main__" in content:
            rel = file.relative_to(repo_root()).with_suffix("")
            modules.append(".".join(rel.parts))
    return sorted(set(modules))


def extract_readme_sections() -> list[tuple[str, str]]:
    readme_path = repo_root() / "README.md"
    if not readme_path.exists():
        return []

    text = readme_path.read_text(encoding="utf-8", errors="replace")
    sections: list[tuple[str, str]] = []
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        return [("README", text[:4000])]

    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body[:2000]))
    return sections


def split_extra_args(extra: str) -> list[str]:
    if not extra.strip():
        return []
    return shlex.split(extra, posix=False)


def append_arg(cmd: list[str], key: str, value: str | int | float | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([f"--{key}", text])


def append_flag(cmd: list[str], key: str, enabled: bool) -> None:
    if enabled:
        cmd.append(f"--{key}")
