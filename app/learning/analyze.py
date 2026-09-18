"""CLI tool: python -m app.learning.analyze"""

import argparse
from app.learning.versioning import LearningCoordinator


def main():
    parser = argparse.ArgumentParser(description="Analyze historical conversation data and build style profiles.")
    parser.add_argument("--version", type=str, default=None, help="Explicit version ID (e.g. style_profile_v001)")
    parser.add_argument("--skip-rag", action="store_true", help="Skip local RAG semantic embedding")
    args = parser.parse_args()

    coordinator = LearningCoordinator()
    print("Running historical intelligence analysis...")
    version_id = coordinator.run_analysis(version_id=args.version, index_rag=not args.skip_rag)
    print(f"\n[Analysis Complete] Generated profile version: {version_id}")


if __name__ == "__main__":
    main()
