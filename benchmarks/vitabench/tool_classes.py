"""Extract VitaBench's official @is_tool read/write labels.

The runtime classification is driven by this table, not by hand-written
allowlists: a tool marked ``ToolType.WRITE`` is a mutation (gated by Strong);
anything else (READ / THINK / GENERIC) is a read and commits unverified.

``scan(vita_root)`` returns ``{tool_name: "read"|"write"|"think"|"generic"}``
for every ``@is_tool``-decorated function in the checkout. ``mutates_map``
collapses that to the gating map ``{tool_name: bool}`` where only WRITE gates.
``report`` produces the startup line the run driver prints (counts plus the
names of tools whose label is neither WRITE nor READ).
"""

from __future__ import annotations

import re
from pathlib import Path

_TOOL_RE = re.compile(
    r"@is_tool\(\s*(?:tool_type\s*=\s*)?ToolType\.(\w+)[^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)"
)


def scan(vita_root: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    src = Path(vita_root) / "src" / "vita"
    for path in src.rglob("*.py"):
        for match in _TOOL_RE.finditer(path.read_text(encoding="utf-8")):
            tool_type, name = match.groups()
            table[name] = tool_type.lower()
    return table


def mutates_map(vita_root: Path) -> dict[str, bool]:
    return {name: t == "write" for name, t in scan(vita_root).items()}


def report(table: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    other: list[str] = []
    for name, t in sorted(table.items()):
        counts[t] = counts.get(t, 0) + 1
        if t not in ("write", "read"):
            other.append(f"{name}({t})")
    counts_s = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    tail = f" non-read/write=[{', '.join(other)}]" if other else ""
    return f"vitabench tools total={len(table)} {counts_s}{tail}"
