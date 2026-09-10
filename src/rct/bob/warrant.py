from __future__ import annotations

from dataclasses import dataclass
import copy

from .types import ProbeAcquisitionWarrant


class WarrantError(RuntimeError):
    pass


@dataclass
class _WarrantState:
    warrant: ProbeAcquisitionWarrant
    consumed: bool = False


class ProbeWarrantRegistry:
    def __init__(self):
        self._entries: dict[str, _WarrantState] = {}
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"W{self._counter:06d}"

    def add(self, warrant: ProbeAcquisitionWarrant) -> None:
        if warrant.warrant_id in self._entries:
            raise WarrantError(f"duplicate warrant {warrant.warrant_id}")
        self._entries[warrant.warrant_id] = _WarrantState(copy.deepcopy(warrant), False)
        if warrant.warrant_id.startswith("W") and warrant.warrant_id[1:].isdigit():
            self._counter = max(self._counter, int(warrant.warrant_id[1:]))

    def get(self, warrant_id: str) -> ProbeAcquisitionWarrant:
        if warrant_id not in self._entries:
            raise WarrantError(f"unknown warrant {warrant_id}")
        return copy.deepcopy(self._entries[warrant_id].warrant)

    def consumed(self, warrant_id: str) -> bool:
        if warrant_id not in self._entries:
            raise WarrantError(f"unknown warrant {warrant_id}")
        return self._entries[warrant_id].consumed

    def is_usable(self, warrant_id: str, current_step: int) -> bool:
        entry = self._entries.get(warrant_id)
        if entry is None or entry.consumed:
            return False
        w = entry.warrant
        return int(w.created_step) <= int(current_step) <= int(w.expires_step)

    def consume(self, warrant_id: str) -> None:
        if warrant_id not in self._entries:
            raise WarrantError(f"unknown warrant {warrant_id}")
        entry = self._entries[warrant_id]
        if entry.consumed:
            raise WarrantError(f"already consumed {warrant_id}")
        entry.consumed = True

    def items(self) -> tuple[ProbeAcquisitionWarrant, ...]:
        return tuple(copy.deepcopy(x.warrant) for _, x in sorted(self._entries.items()))
