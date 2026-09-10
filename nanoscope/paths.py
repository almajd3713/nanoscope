"""Filesystem paths in YAML configs, including the @/ project-root alias."""

from __future__ import annotations

from pathlib import Path


def project_root(config_path: str | Path) -> Path:
    """Find the nearest project marker above the YAML, independent of the process cwd."""
    directory = Path(config_path).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    raise ValueError(
        f"cannot resolve @/ from {config_path}: no parent contains pyproject.toml or .git; "
        "place the config inside the project or use an absolute/relative path"
    )


def config_path(value: str, source: str | Path) -> Path:
    """Resolve @/ from the project root, other relative paths from the source YAML."""
    if not isinstance(value, str) or not value:
        raise ValueError("paths must be non-empty strings")
    if value.startswith("@"):
        if value != "@" and (not value.startswith("@/") or value.startswith("@//")):
            raise ValueError("project-root paths must use @ or @/path (quoted in YAML)")
        suffix = "" if value == "@" else value[2:]
        return (project_root(source) / suffix).resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(source).parent / path).resolve()
