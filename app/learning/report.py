"""CLI tool: python -m app.learning.report"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.config import get_settings
from app.learning.dataset_scanner import DatasetScanner
from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository


def format_qualitative_tier(val: float) -> str:
    """Format float into qualitative descriptor for report."""
    if val >= 0.75:
        return "VERY HIGH"
    elif val >= 0.50:
        return "HIGH"
    elif val >= 0.25:
        return "MEDIUM"
    elif val >= 0.10:
        return "LOW"
    return "VERY LOW"


def generate_report_text(repo: HistoricalRepository, manifest_path: Path | None = None) -> str:
    """Compose the structured privacy-preserving historical intelligence report."""
    scanner = DatasetScanner()
    manifest = scanner.scan()

    op_style = repo.get_operator_style()
    active_version = repo.get_active_version()
    contacts = repo.list_all_contact_styles()
    reels = repo.get_all_reels()
    memories = repo.get_top_memories(limit=100)
    high_conf_memories = [m for m in memories if m.get("confidence", 0) >= 0.75]

    lines = [
        "HISTORICAL LEARNING REPORT",
        "==========================",
        "",
        "Dataset",
        "-------",
        f"Conversations: {manifest.valid_conversations}",
        f"Messages: {manifest.total_messages:,}",
    ]

    if manifest.date_range:
        lines.append(f"Date range: {manifest.date_range[0]} \u2192 {manifest.date_range[1]}")
    else:
        lines.append("Date range: None")

    lines.append(f"Files: {manifest.files_discovered}")
    lines.append("")

    lines.extend([
        "OPERATOR STYLE",
        "--------------",
    ])
    if op_style:
        lines.append(f"Hinglish: {format_qualitative_tier(op_style.get('hinglish_ratio', 0))}")
        lines.append(f"Lowercase: {format_qualitative_tier(op_style.get('lowercase_ratio', 0))}")
        short_msg_score = 1.0 if op_style.get('avg_words_per_message', 10) < 6 else 0.4
        lines.append(f"Short messages: {format_qualitative_tier(short_msg_score)}")
        lines.append(f"Emoji: {format_qualitative_tier(op_style.get('emoji_density', 0))}")
        lines.append(f"Burstiness: {format_qualitative_tier(op_style.get('burst_message_ratio', 0))}")
        lines.append(f"Ending period avoidance: {format_qualitative_tier(1.0 - op_style.get('ending_period_ratio', 0))}")
    else:
        lines.append("[No operator profile generated yet. Run 'python -m app.learning.analyze']")
    lines.append("")

    lines.extend([
        "CONTACTS",
        "--------",
        f"{len(contacts)} conversation profiles generated",
        "",
    ])

    sent_reels = sum(1 for r in reels if r.get("sender_is_operator"))
    recv_reels = len(reels) - sent_reels
    lines.extend([
        "REELS",
        "-----",
        f"Total Shared: {len(reels)}",
        f"Sent by Operator: {sent_reels}",
        f"Received by Operator: {recv_reels}",
        "",
    ])

    lines.extend([
        "MEMORIES",
        "--------",
        f"Candidates: {len(memories)}",
        f"High confidence: {len(high_conf_memories)}",
        "",
    ])

    ver_label = active_version.get("version_id", "None") if active_version else "None"
    conf_label = f"{op_style.get('confidence', 0.0):.2f}" if op_style else "0.00"
    lines.extend([
        "STYLE PROFILE",
        "-------------",
        f"Version: {ver_label}",
        f"Confidence: {conf_label}",
        f"Algorithm: 2.0.0-historical-intelligence",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate human-readable historical learning report.")
    parser.add_argument("--save", action="store_true", default=True, help="Save report to reports directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom file path to save the report (markdown or txt)")
    args = parser.parse_args()

    settings = get_settings()
    repo = HistoricalRepository(get_db_manager())
    report_text = generate_report_text(repo)

    print("\n" + report_text + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        print(f"[Report saved to {out_path}]")
    elif args.save:
        reports_dir = settings.resolved_historical_base_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"historical_report_{now_str}.txt"
        report_file.write_text(report_text, encoding="utf-8")
        print(f"[Report saved to {report_file}]")


if __name__ == "__main__":
    main()
