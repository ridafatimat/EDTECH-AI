from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from .common import RunRequest


class RenderVisualsRequest(RunRequest):
    """Render one semantic visual family for the current Notebook 06 handoff."""

    model_config = ConfigDict(extra="forbid")

    quiz_mode: Literal["complete_quiz", "fill_shortfall"]
