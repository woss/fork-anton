"""Datasource registry — parses datasources.md engine definitions.

Reads YAML blocks from the built-in datasources.md (project root) and
merges user overrides from ~/.anton/datasources.md on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


@dataclass
class DatasourceField:
    name: str
    required: bool = True
    secret: bool = False
    description: str = ""
    default: str = ""


@dataclass
class DatasourceEngine:
    engine: str
    display_name: str
    pip: str = ""
    name_from: Union[str, list[str]] = ""
    fields: list[DatasourceField] = field(default_factory=list)
    test_snippet: str = ""


#Parse the file for engine defintions and extract the YAML blocks.
_YAML_BLOCK_RE = re.compile(
    r"^##\s+(.+?)\s*$\n(.*?)^```yaml\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)


def _parse_file(path: Path) -> dict[str, DatasourceEngine]:
    """Extract engine definitions from a datasources.md file."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    engines: dict[str, DatasourceEngine] = {}

    for match in _YAML_BLOCK_RE.finditer(text):
        yaml_text = match.group(3)
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict) or "engine" not in data:
            continue

        raw_fields = data.get("fields", []) or []
        parsed_fields: list[DatasourceField] = []
        for f in raw_fields:
            if not isinstance(f, dict):
                continue
            parsed_fields.append(DatasourceField(
                name=f.get("name", ""),
                required=bool(f.get("required", True)),
                secret=bool(f.get("secret", False)),
                description=f.get("description", ""),
                default=str(f.get("default", "")),
            ))

        engine_slug = str(data["engine"])
        engines[engine_slug] = DatasourceEngine(
            engine=engine_slug,
            display_name=str(data.get("display_name", engine_slug)),
            pip=str(data.get("pip", "")),
            name_from=data.get("name_from", ""),
            fields=parsed_fields,
            test_snippet=str(data.get("test_snippet", "")),
        )

    return engines


class DatasourceRegistry:
    """Parsed registry of all available data source engines."""

    # Default connection definition 
    _BUILTIN_PATH: Path = Path(__file__).parent.parent / "datasources.md"
    # If user adds new connection
    _USER_PATH: Path = Path("~/.anton/datasources.md").expanduser()

    def __init__(self) -> None:
        self._engines: dict[str, DatasourceEngine] = {}
        self._load()

    def _load(self) -> None:
        self._engines = _parse_file(self._BUILTIN_PATH)
        # Merge user overrides on top
        for slug, engine in _parse_file(self._USER_PATH).items():
            self._engines[slug] = engine

    def get(self, engine_slug: str) -> DatasourceEngine | None:
        """Look up an engine by its slug (e.g. 'postgres')."""
        return self._engines.get(engine_slug)

    def find_by_name(self, display_name: str) -> DatasourceEngine | None:
        """Case-insensitive match on display_name (e.g. 'PostgreSQL')."""
        needle = display_name.strip().lower()
        for engine in self._engines.values():
            if engine.display_name.lower() == needle:
                return engine
            # Also match the slug directly
            if engine.engine.lower() == needle:
                return engine
        return None

    def all_engines(self) -> list[DatasourceEngine]:
        return sorted(self._engines.values(), key=lambda e: e.display_name)

    def derive_name(self, engine_def: DatasourceEngine, credentials: dict[str, str]) -> str:
        """Derive a default connection name from name_from field(s).

        Falls back to the engine slug if the field is missing or empty.
        """
        name_from = engine_def.name_from
        if not name_from:
            return engine_def.engine
        if isinstance(name_from, str):
            return credentials.get(name_from, engine_def.engine)
        parts = [credentials.get(f, "") for f in name_from if credentials.get(f)]
        return "_".join(parts) if parts else engine_def.engine
