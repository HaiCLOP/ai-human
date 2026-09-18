"""CLI tool: python -m app.learning.inspect"""

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository


def main():
    parser = argparse.ArgumentParser(description="Inspect learned operator or contact style profiles.")
    parser.add_argument("--operator", action="store_true", help="Inspect global operator style profile")
    parser.add_argument("--contact", type=str, default=None, help="Inspect contact by display name or contact_id")
    parser.add_argument("--list-contacts", action="store_true", help="List all profiled contacts")
    args = parser.parse_args()

    repo = HistoricalRepository(get_db_manager())

    if args.list_contacts:
        contacts = repo.list_all_contact_styles()
        print(f"\nProfiled Contacts ({len(contacts)} total):")
        print("--------------------------------------------------")
        for c in contacts:
            print(f"ID: {c['contact_id']} | Name: {c['display_name']} | Messages: {c['total_messages']} | Brevity: {c['brevity_level']}")
        return

    if args.contact:
        contact_id = args.contact
        if not contact_id.startswith("contact_"):
            from app.learning.operator_detector import OperatorDetector
            contact_id = OperatorDetector.generate_contact_id(args.contact)

        style = repo.get_contact_style(contact_id)
        rel = repo.get_relationship_style(contact_id)
        humor = repo.get_humor_profile(contact_id)
        mems = repo.get_top_memories(contact_id, limit=5)

        if not style:
            print(f"No profile found for contact '{args.contact}' (ID: {contact_id})")
            return

        print(f"\n--- CONTACT PROFILE: {style.get('display_name')} ({contact_id}) ---")
        print(json.dumps(style, indent=2))
        if rel:
            print("\n--- RELATIONSHIP DYNAMICS ---")
            print(json.dumps(rel, indent=2))
        if humor:
            print("\n--- HUMOR DYNAMICS ---")
            print(json.dumps(humor, indent=2))
        if mems:
            print(f"\n--- TOP MEMORIES ({len(mems)}) ---")
            for m in mems:
                print(f"[{m.get('category')}] {m.get('statement')} (conf: {m.get('confidence')})")
        return

    # Default to operator style
    op = repo.get_operator_style()
    if not op:
        print("No operator style profile found. Run 'python -m app.learning.analyze' first.")
        return

    print("\n--- OPERATOR STYLE PROFILE ---")
    print(json.dumps(op, indent=2))


if __name__ == "__main__":
    main()
