"""Call tree visualization for recursive dialectical traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import CallTreeNode


def call_tree_to_json(tree: CallTreeNode) -> str:
    """Serialize a call tree to JSON."""
    return tree.model_dump_json(indent=2)


def call_tree_to_ascii(tree: CallTreeNode, indent: int = 0) -> str:
    """Render a call tree as an ASCII tree for terminal display."""
    lines: list[str] = []
    prefix = "  " * indent
    connector = "├─ " if indent > 0 else ""

    label = _node_label(tree)
    lines.append(f"{prefix}{connector}{label}")

    for i, child in enumerate(tree.children):
        is_last = i == len(tree.children) - 1
        child_prefix = "└─ " if is_last else "├─ "
        child_lines = _render_child(child, indent + 1, child_prefix)
        lines.extend(child_lines)

    return "\n".join(lines)


def _node_label(node: CallTreeNode) -> str:
    """Build a concise label for a tree node."""
    parts = [f"[{node.node_type}]"]
    if node.role:
        parts.append(node.role)
    if node.model:
        parts.append(f"({node.model})")
    if node.latency_ms > 0:
        parts.append(f"{node.latency_ms:.0f}ms")
    if node.output_summary:
        summary = node.output_summary[:60].replace("\n", " ")
        parts.append(f"→ {summary}")
    return " ".join(parts)


def _render_child(node: CallTreeNode, indent: int, connector: str) -> list[str]:
    """Render a child node and its descendants."""
    lines: list[str] = []
    prefix = "  " * indent
    label = _node_label(node)
    lines.append(f"{prefix}{connector}{label}")

    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        child_connector = "└─ " if is_last else "├─ "
        child_lines = _render_child(child, indent + 1, child_connector)
        lines.extend(child_lines)

    return lines


def save_call_tree(tree: CallTreeNode, path: str) -> None:
    """Save a call tree to a JSON file."""
    with open(path, "w") as f:
        f.write(call_tree_to_json(tree))
