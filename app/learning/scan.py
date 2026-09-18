"""CLI tool: python -m app.learning.scan"""

import argparse
from pathlib import Path

from app.learning.dataset_scanner import DatasetScanner


def main():
    parser = argparse.ArgumentParser(description="Scan and validate Instagram conversation export dataset.")
    parser.add_argument("--path", type=str, default=None, help="Custom dataset directory path")
    parser.add_argument("--save-manifest", action="store_true", default=True, help="Save manifest to processed dir")
    args = parser.parse_args()

    scanner = DatasetScanner(args.path)
    manifest = scanner.scan()

    if args.save_manifest and manifest.files_discovered > 0:
        manifest_path = scanner.save_manifest(manifest)
        print(f"[Manifest saved to {manifest_path}]")

    print("\n" + DatasetScanner.format_manifest_report(manifest))


if __name__ == "__main__":
    main()
