"""Browser agent data models and DOM observation structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DOMMessage:
    """Represents a message bubble extracted from the Instagram Web DOM."""

    sender_handle: str
    sender_type: Literal["USER", "CHARACTER"]
    text: str
    element_index: int


@dataclass
class BrowserDiagnosticSnapshot:
    """Diagnostic evidence captured upon errors or security challenges."""

    timestamp_iso: str
    url: str
    screenshot_path: str
    dom_html_path: str
    trigger_reason: str
