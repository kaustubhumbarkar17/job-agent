"""Checkpoints — human-in-the-loop gate abstraction.

Three implementations:
  InteractiveCheckpoint  – questionary/rich prompts (production)
  ScriptedCheckpoint     – reads from a pre-defined list (tests)
  AutoCheckpoint         – auto-selects all (--yes flag)
"""
from __future__ import annotations

import abc
from typing import Any


class Checkpoint(abc.ABC):
    """Abstract checkpoint with select and confirm."""

    @abc.abstractmethod
    def select(
        self,
        items: list[dict[str, Any]],
        preselected: list[int] | None = None,
        label: str = "Select",
    ) -> list[int]:
        """Present items to the user and return selected IDs.

        Args:
            items: list of {"id": int, "label": str, ...} dicts.
            preselected: optional list of IDs already chosen.
            label: prompt label shown to the user.

        Returns:
            List of selected item IDs.
        """

    @abc.abstractmethod
    def confirm(self, message: str) -> bool:
        """Ask the user to confirm an action.

        Args:
            message: The confirmation prompt text.

        Returns:
            True if confirmed, False otherwise.
        """


class InteractiveCheckpoint(Checkpoint):
    """Real interactive checkpoint using questionary."""

    def __init__(self) -> None:
        import questionary  # lazy import to avoid CLI freeze

        self._questionary = questionary

    def select(
        self,
        items: list[dict[str, Any]],
        preselected: list[int] | None = None,
        label: str = "Select",
    ) -> list[int]:
        if not items:
            return []

        defaults = set(preselected or [])
        choices = [
            self._questionary.Choice(item["label"], value=item["id"], checked=item["id"] in defaults)
            for item in items
        ]
        chosen = self._questionary.checkbox(
            f"{label} (Space to select, Enter to confirm):",
            choices=choices,
        ).ask()

        if chosen is None:
            print("  Cancelled.")
            return []

        return chosen

    def confirm(self, message: str) -> bool:
        return bool(self._questionary.confirm(message, default=False).ask())


class ScriptedCheckpoint(Checkpoint):
    """Scripted checkpoint for tests. Reads from a pre-defined list."""

    def __init__(self, selections: list[list[int]] | None = None) -> None:
        """
        Args:
            selections: A list of ID lists, one per select() call.
                        E.g. [[1, 3], [2]] means first select returns [1,3],
                        second returns [2].
        """
        self._selections = selections or []
        self._call_index = 0

    def select(
        self,
        items: list[dict[str, Any]],
        preselected: list[int] | None = None,
        label: str = "Select",
    ) -> list[int]:
        available_ids = {item["id"] for item in items}
        if self._call_index < len(self._selections):
            result = [
                sid for sid in self._selections[self._call_index]
                if sid in available_ids
            ]
            self._call_index += 1
            return result
        # Default: select all remaining items
        return [item["id"] for item in items]

    def confirm(self, message: str) -> bool:
        return True  # Always confirm in scripted mode


class AutoCheckpoint(Checkpoint):
    """Auto-selects all items (--yes flag)."""

    def select(
        self,
        items: list[dict[str, Any]],
        preselected: list[int] | None = None,
        label: str = "Select",
    ) -> list[int]:
        return [item["id"] for item in items]

    def confirm(self, message: str) -> bool:
        return True  # Always confirm
