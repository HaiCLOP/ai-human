"""Dataset scanner and manifest generator for historical Instagram exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.learning.models import DatasetManifest, MessageType
from app.learning.normalizer import InstagramNormalizer

logger = get_logger("learning.scanner")


class DatasetScanner:
    """Scans and validates Instagram conversation export directories."""

    def __init__(self, dataset_path: Path | str | None = None):
        settings = get_settings()
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = settings.resolved_historical_dataset_path

    def compute_dataset_hash(self, files: list[Path]) -> str:
        """Compute a deterministic SHA-256 hash of the dataset based on file paths and contents."""
        hasher = hashlib.sha256()
        for f in sorted(files, key=lambda p: str(p)):
            try:
                rel_path = f.relative_to(self.dataset_path)
            except ValueError:
                rel_path = f.name
            hasher.update(str(rel_path).encode("utf-8"))
            stat = f.stat()
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    def scan(self) -> DatasetManifest:
        """Discover, validate, and compute manifest for all JSON files in the dataset directory."""
        if not self.dataset_path.exists():
            logger.warning("scanner.dataset_dir_missing", path=str(self.dataset_path))
            return DatasetManifest()

        discovered_files = list(self.dataset_path.rglob("*.json"))
        # Exclude internal manifest/profile files if scanner points to base directory
        candidate_files = [f for f in discovered_files if "manifest" not in f.name and "processed" not in str(f)]

        manifest = DatasetManifest(
            files_discovered=len(candidate_files),
        )

        valid_files: list[Path] = []
        min_ts: int | None = None
        max_ts: int | None = None
        type_counts: dict[str, int] = {t.value: 0 for t in MessageType}

        for path in candidate_files:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except Exception as e:
                manifest.invalid_files += 1
                manifest.invalid_file_reasons[path.name] = f"Invalid JSON: {e}"
                continue

            if not isinstance(data, dict) or "messages" not in data:
                manifest.invalid_files += 1
                manifest.invalid_file_reasons[path.name] = "Missing 'messages' array in root object"
                continue

            messages = data.get("messages", [])
            if not isinstance(messages, list):
                manifest.invalid_files += 1
                manifest.invalid_file_reasons[path.name] = "'messages' field is not a list"
                continue

            manifest.valid_conversations += 1
            manifest.total_messages += len(messages)
            valid_files.append(path)

            for m in messages:
                if not isinstance(m, dict):
                    continue
                mtype = InstagramNormalizer.classify_message_type(m)
                type_counts[mtype.value] = type_counts.get(mtype.value, 0) + 1

                ts = m.get("timestamp_ms")
                if isinstance(ts, (int, float)) and ts > 0:
                    ts_int = int(ts)
                    if min_ts is None or ts_int < min_ts:
                        min_ts = ts_int
                    if max_ts is None or ts_int > max_ts:
                        max_ts = ts_int

        manifest.type_counts = type_counts
        manifest.dataset_hash = self.compute_dataset_hash(valid_files)

        if min_ts and max_ts:
            from datetime import datetime, timezone
            dt_start = datetime.fromtimestamp(min_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            dt_end = datetime.fromtimestamp(max_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            manifest.date_range = (dt_start, dt_end)

        return manifest

    def save_manifest(self, manifest: DatasetManifest, target_path: Path | None = None) -> Path:
        """Persist manifest to processed directory for quick change detection."""
        settings = get_settings()
        dest_dir = target_path.parent if target_path else settings.resolved_historical_base_path / "processed"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_path or dest_dir / "manifest.json"

        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info("scanner.manifest_saved", path=str(dest_file), total_messages=manifest.total_messages)
        return dest_file

    @staticmethod
    def format_manifest_report(manifest: DatasetManifest) -> str:
        """Generate human-readable summary matching production spec."""
        tc = manifest.type_counts
        lines = [
            "Instagram Historical Dataset",
            "============================",
            "",
            f"Files discovered: {manifest.files_discovered}",
            f"Valid conversations: {manifest.valid_conversations}",
            f"Invalid files: {manifest.invalid_files}",
            "",
            f"Total messages: {manifest.total_messages:,}",
            f"Text messages: {tc.get('TEXT', 0):,}",
            f"Reels: {tc.get('REEL', 0):,}",
            f"Stories: {tc.get('STORY', 0):,}",
            f"Photos: {tc.get('PHOTO', 0):,}",
            f"Videos: {tc.get('VIDEO', 0):,}",
            f"Audio: {tc.get('AUDIO', 0):,}",
            f"Reactions: {tc.get('REACTION', 0):,}",
            f"Calls: {tc.get('CALL', 0):,}",
            f"Other: {tc.get('ATTACHMENT', 0) + tc.get('UNKNOWN', 0):,}",
            "",
        ]

        if manifest.date_range:
            lines.extend([
                "Date range:",
                f"{manifest.date_range[0]} -> {manifest.date_range[1]}",
            ])
        else:
            lines.append("Date range: None")

        if manifest.invalid_file_reasons:
            lines.extend(["", "Invalid File Details:"])
            for fname, reason in manifest.invalid_file_reasons.items():
                lines.append(f"  - {fname}: {reason}")

        return "\n".join(lines)
