"""Small typed models shared by the local evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputRecord:
    """One canonical-ID answer record used by alignment and scoring."""

    id: str
    answer: str


@dataclass(frozen=True)
class AlignedRecord:
    """A reference and prediction paired by canonical ID."""

    id: str
    reference: str
    prediction: str
