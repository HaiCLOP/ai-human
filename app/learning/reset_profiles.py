"""CLI tool: python -m app.learning.reset_profiles"""

from app.learning.versioning import LearningCoordinator


def main():
    coordinator = LearningCoordinator()
    print("Resetting derived style profiles, observations, humor, and memories...")
    coordinator.reset_profiles()
    print("[Reset Complete] Derived intelligence profiles cleared. Raw message history preserved.")


if __name__ == "__main__":
    main()
