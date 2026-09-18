"""CLI tool: python -m app.learning.rebuild"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.learning.versioning import LearningCoordinator


def main():
    parser = argparse.ArgumentParser(description="Full end-to-end rebuild of historical intelligence models from source JSONs.")
    parser.add_argument("--path", type=str, default=None, help="Custom dataset directory path")
    args = parser.parse_args()

    coordinator = LearningCoordinator()
    print("Initiating full rebuild from source conversations...")
    version_id = coordinator.full_rebuild(dataset_path=args.path)
    print(f"\n[Rebuild Complete] Re-generated intelligence models under version: {version_id}")


if __name__ == "__main__":
    main()
