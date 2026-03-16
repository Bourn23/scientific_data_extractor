"""
Structured JSON Lines logger for the PageIndex extraction pipeline.
Writes one event per line to extraction_log.jsonl so the verifier agent
can read and reason about failures without parsing stdout.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Appends structured JSON events to a .jsonl log file.

    Each event contains:
        timestamp   : ISO-ish wall-clock string
        asset_type  : "text" | "image" | "table" | "pipeline"
        asset_id    : section title, image filename, table caption, or doc name
        status      : "started" | "success" | "empty" | "failed" | "completed"
        error_type  : "timeout" | "model_overloaded" | "exception" | None
        error_msg   : raw exception string, or None
        retry_count : how many attempts were made before this outcome
        + any extra fields passed via the `extra` kwarg (e.g. image_path, section_content)
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        # Clear / create the file at pipeline start
        self.log_path.write_text("", encoding="utf-8")

    def log(
        self,
        asset_type: str,
        asset_id: str,
        status: str,
        error_type: Optional[str] = None,
        error_msg: Optional[str] = None,
        retry_count: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "asset_type": asset_type,
            "asset_id": asset_id,
            "status": status,
            "error_type": error_type,
            "error_msg": error_msg,
            "retry_count": retry_count,
        }
        if extra:
            event.update(extra)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
