"""Load and split an Agent Behavior spec (agentbehavior.dev).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SPEC_PATH = Path(".agents/behaviors/voice-call-conduct/BEHAVIOR.md")

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
_HEADING = re.compile(r"^##\s+(.+?)\s*$")


@dataclass(frozen=True)
class BehaviorSection:
    """One `## ...` section: the rubric handed to the judge, verbatim."""

    title: str
    slug: str
    body: str


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    description: str
    body: str
    sections: list[BehaviorSection]
    location: str


def _slugify(title: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", title.lower()))


def _split_sections(body: str) -> list[BehaviorSection]:
    sections: list[BehaviorSection] = []
    title: str | None = None
    lines: list[str] = []

    def finalize() -> None:
        if title is not None:
            sections.append(
                BehaviorSection(
                    title=title, slug=_slugify(title), body="\n".join(lines).strip()
                )
            )

    for line in body.split("\n"):
        match = _HEADING.match(line)
        if match:
            finalize()
            title = match.group(1)
            lines = [line]
        elif title is not None:
            lines.append(line)

    finalize()
    return sections


def load_behavior_spec(path: Path | str = DEFAULT_SPEC_PATH) -> BehaviorSpec:
    """Parse a BEHAVIOR.md into its frontmatter and judged sections."""
    location = Path(path).resolve()
    raw = location.read_text(encoding="utf-8")

    match = _FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"Spec at {location} has no YAML frontmatter.")
    data = yaml.safe_load(match.group(1)) or {}
    content = match.group(2)

    name = data.get("name") or ""
    description = data.get("description") or ""
    if not name or not description:
        raise ValueError(
            f"Spec at {location} is missing a name or description in its frontmatter."
        )

    sections = _split_sections(content)
    if not sections:
        raise ValueError(f'Spec at {location} has no "## " behavior sections to judge.')

    return BehaviorSpec(
        name=name,
        description=description,
        body=content.strip(),
        sections=sections,
        location=str(location),
    )
