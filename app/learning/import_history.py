"""CLI tool: python -m app.learning.import_history"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.learning.versioning import LearningCoordinator


def main():
    parser = argparse.ArgumentParser(description="Import Instagram JSON exports into local SQLite database.")
    parser.add_argument("--path", type=str, default=None, help="Custom dataset directory path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conversations to process")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate without writing to database")
    args = parser.parse_args()

    coordinator = LearningCoordinator()

    if args.dry_run:
        from app.learning.dataset_scanner import DatasetScanner
        scanner = DatasetScanner(args.path)
        manifest = scanner.scan()
        print("[DRY RUN] Dataset scan completed:")
        print(DatasetScanner.format_manifest_report(manifest))
        return

    print("Importing conversations into SQLite...")
    conv_count, msg_count = coordinator.import_dataset(dataset_path=args.path, limit=args.limit)
    summary = coordinator.last_import_summary
    tc = summary.type_counts

    print("\n==================================================")
    print("Historical Dataset Import Report")
    print("==================================================")
    print(f"Files discovered: {summary.files_discovered}")
    print(f"Files successfully processed: {summary.files_processed}")
    print(f"Files rejected: {len(summary.files_rejected)}")
    print(f"Messages imported: {summary.messages_imported:,}")
    print(f"Text messages: {tc.get('TEXT', 0):,}")
    print(f"Reels: {tc.get('REEL', 0):,}")
    print(f"Stories: {tc.get('STORY', 0):,}")
    print(f"Photos: {tc.get('PHOTO', 0):,}")
    print(f"Videos: {tc.get('VIDEO', 0):,}")
    print(f"Audio: {tc.get('AUDIO', 0):,}")
    print(f"Reactions: {tc.get('REACTION', 0):,}")
    print(f"Calls: {tc.get('CALL', 0):,}")
    print(f"Other: {tc.get('ATTACHMENT', 0) + tc.get('UNKNOWN', 0):,}")

    if summary.files_rejected:
        print("\nRejected Files:")
        for fname, reason in summary.files_rejected.items():
            print(f"  - {fname}: {reason}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
