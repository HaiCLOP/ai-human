"""Main application entrypoint for the Fictional AI Character Instagram Browser Agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.ai.persona import load_character_profile
from app.ai.router import get_llm_provider
from app.browser.instagram import InstagramBrowserAgent
from app.browser.session import BrowserSessionManager
from app.conversation.manager import ConversationManager
from app.core.config import get_settings
from app.core.exceptions import SecurityChallengeException
from app.core.logging import get_logger, setup_logging
from app.rag.indexer import KnowledgeIndexer
from app.rag.retriever import LocalRAGRetriever
from app.storage.database import get_db_manager

logger = get_logger("main")


async def run_setup_login() -> None:
    """Run operator interactive login mode."""
    session_mgr = BrowserSessionManager()
    await session_mgr.run_interactive_setup()


async def run_reindex_rag() -> None:
    """Index or re-index knowledge/ directory into local vector storage."""
    db = get_db_manager()
    db.initialize_schema()
    indexer = KnowledgeIndexer(db=db)
    knowledge_dir = Path("knowledge")
    count = indexer.index_directory(knowledge_dir)
    print(f"\nSuccessfully indexed {count} chunks from {knowledge_dir} into SQLite vector storage.\n")


async def run_dry_run_console(force_available: bool = False) -> None:
    """Interactive console simulation of character conversation without browser."""
    print("\n" + "=" * 70)
    print("  FICTIONAL AI CHARACTER — CONSOLE DRY RUN MODE")
    print("  Type a message and press Enter to chat with the character.")
    if force_available:
        print("  [Mode: Force Available (Bypassing Routine Schedule)]")
    else:
        print("  [Mode: Natural Routine (Pass --force-available to bypass schedule)]")
    print("  Type 'exit' or 'quit' to end.")
    print("=" * 70 + "\n")

    db = get_db_manager()
    db.initialize_schema()

    # Index RAG if empty
    rag_retriever = LocalRAGRetriever(db=db)
    rag_retriever.reload_index()
    if len(rag_retriever._chunks) == 0:
        indexer = KnowledgeIndexer(db=db)
        indexer.index_directory("knowledge")
        rag_retriever.reload_index()

    profile = load_character_profile()
    conv_mgr = ConversationManager(
        db=db,
        character_profile=profile,
        rag_retriever=rag_retriever,
    )

    conv_id = "console_dry_run_thread"
    user_handle = "@ConsoleUser"

    while True:
        try:
            user_input = await asyncio.to_thread(input, f"{user_handle}: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            reply = await conv_mgr.handle_incoming_message(
                conversation_id=conv_id,
                sender_handle=user_handle,
                message_text=user_input,
                force_available=force_available,
            )

            if reply:
                print(f"{profile.identity.name}: {reply}\n")
            else:
                from datetime import datetime, timezone
                avail_state, current_activity = conv_mgr.routine_mgr.resolve_availability(datetime.now(timezone.utc))
                if not force_available and avail_state.value != "AVAILABLE":
                    print(f"[{profile.identity.name} is currently away ({current_activity.activity}, until {current_activity.end} UTC). Message queued. Run with --force-available to chat anytime.]\n")
                else:
                    print(f"[{profile.identity.name} chose silence or output was suppressed]\n")

        except (KeyboardInterrupt, EOFError):
            break


async def run_agent(
    target_username: str | None = None,
    custom_message: str | None = None,
    force_available: bool = False,
    once: bool = False,
) -> int:
    """Main autonomous browser monitoring loop with proactive outreach support."""
    settings = get_settings()
    db = get_db_manager()
    db.initialize_schema()

    # Ensure local RAG is populated
    rag_retriever = LocalRAGRetriever(db=db)
    rag_retriever.reload_index()
    if len(rag_retriever._chunks) == 0:
        logger.info("main.seeding_rag_knowledge")
        indexer = KnowledgeIndexer(db=db)
        indexer.index_directory("knowledge")
        rag_retriever.reload_index()

    session_mgr = BrowserSessionManager()
    profile = load_character_profile()

    conv_mgr = ConversationManager(
        db=db,
        character_profile=profile,
        rag_retriever=rag_retriever,
    )
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    logger.info("main.launching_browser_agent", character=profile.identity.name)

    raw_target = (target_username.strip() if target_username and target_username.strip() else "") or settings.TARGET_USERNAME
    clean_target = raw_target.lstrip("@").strip() if raw_target else ""

    playwright, context = await session_mgr.launch_persistent_context()
    try:
        page = context.pages[0] if context.pages else await context.new_page()

        if clean_target:
            target_url = f"https://www.instagram.com/{clean_target}/"
        elif settings.TARGET_THREAD_ID:
            target_url = f"https://www.instagram.com/direct/t/{settings.TARGET_THREAD_ID}/"
        else:
            target_url = "https://www.instagram.com/direct/inbox/"

        logger.info("main.navigating_to_instagram", url=target_url)
        await page.goto(target_url, wait_until="domcontentloaded", timeout=settings.BROWSER_NAVIGATION_TIMEOUT_MS)

        # Immediate checkpoint check
        await agent.check_security_checkpoints(page)

        # Detect unauthenticated redirection to login
        if "accounts/login" in page.url:
            logger.warning("main.login_required")
            print("\n" + "!" * 70)
            print("  INSTAGRAM SESSION NOT FOUND: LOGIN REQUIRED")
            print("  Please authenticate manually once to save your persistent session:")
            print("  Run: .venv\\Scripts\\python.exe -m app.main --setup-login")
            print("!" * 70 + "\n")
            return 2

        if clean_target:
            logger.info("main.initiating_outreach", target=clean_target)
            sent_text = await agent.text_user(
                page=page,
                username=clean_target,
                custom_message=custom_message,
                force_available=force_available,
            )
            if sent_text:
                print(f"\n[Sent DM to @{clean_target}]: {sent_text}\n")
            else:
                print(f"\n[Unable to send message to @{clean_target}. Check character availability or logs.]\n")

            if once:
                logger.info("main.once_mode_complete")
                return 0

            # Continue monitoring thread for replies from target user
            conv_id = f"thread_{clean_target}"
            sender_handle = f"@{clean_target}"
            await agent.monitor_thread(
                page=page,
                conversation_id=conv_id,
                sender_handle=sender_handle,
                force_available=force_available,
            )
        else:
            conversation_id = settings.TARGET_THREAD_ID or "active_inbox_thread"
            sender_handle = "DirectParticipant"
            await agent.monitor_thread(
                page=page,
                conversation_id=conversation_id,
                sender_handle=sender_handle,
                force_available=force_available,
            )

        return 0

    except SecurityChallengeException as sce:
        logger.critical("main.emergency_halt_challenge_detected", error=str(sce))
        print("\n" + "!" * 70)
        print("  CRITICAL: INSTAGRAM SECURITY CHALLENGE DETECTED")
        print("  Automated execution halted immediately.")
        print("  Never attempt circumvention.")
        print("  Run 'python -m app.main --setup-login' to verify manually.")
        print("!" * 70 + "\n")
        return 101

    except Exception as e:
        logger.error("main.unhandled_exception", error=str(e))
        return 1

    finally:
        await context.close()
        await playwright.stop()
        logger.info("main.browser_closed_cleanly")


async def run_scrape_chat(target_username: str, scrolls: int = 12) -> int:
    """Scrape message history from target DM thread using persistent browser session."""
    settings = get_settings()
    db = get_db_manager()
    db.initialize_schema()

    session_mgr = BrowserSessionManager()
    profile = load_character_profile()
    conv_mgr = ConversationManager(db=db, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    clean_target = target_username.lstrip("@").strip()
    playwright, context = await session_mgr.launch_persistent_context()
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        await agent.scrape_chat_history(page, clean_target, max_scrolls=scrolls)
        return 0
    except SecurityChallengeException as sce:
        logger.critical("main.emergency_halt_challenge_detected", error=str(sce))
        return 101
    except Exception as e:
        logger.error("main.scrape_chat_error", error=str(e))
        return 1
    finally:
        await context.close()
        await playwright.stop()


def main() -> None:
    """CLI entrypoint dispatcher."""
    parser = argparse.ArgumentParser(
        description="Fictional AI Character - Instagram Browser Agent"
    )
    parser.add_argument(
        "--setup-login",
        action="store_true",
        help="Open visible browser to log in and persist session manually.",
    )
    parser.add_argument(
        "--reindex-rag",
        action="store_true",
        help="Parse and index markdown knowledge files in knowledge/ into SQLite vector storage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run interactive character chat session in terminal without opening Instagram.",
    )
    parser.add_argument(
        "--scrape-chat",
        type=str,
        default=None,
        metavar="USERNAME",
        help="Scrape and analyze previous chat messages from a user (e.g. --scrape-chat @haiclop).",
    )
    parser.add_argument(
        "--scrolls",
        type=int,
        default=12,
        help="Number of scroll steps to load older messages during scraping (default: 12).",
    )
    parser.add_argument(
        "--target",
        "--text-user",
        "--user",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Instagram username to text or monitor (e.g. haiclop or '@haiclop').",
    )
    parser.add_argument(
        "--message",
        "--text",
        type=str,
        default=None,
        help="Optional message content to send. If omitted, character crafts an in-character opener.",
    )
    parser.add_argument(
        "--force-available",
        action="store_true",
        help="Bypass schedule/sleep gating when dispatching operator-requested outreach.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send message and exit without entering continuous inbox monitor loop.",
    )

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_file_path=settings.resolved_log_file_path,
    )

    if args.setup_login:
        asyncio.run(run_setup_login())
    elif args.reindex_rag:
        asyncio.run(run_reindex_rag())
    elif args.dry_run:
        asyncio.run(run_dry_run_console(force_available=args.force_available))
    elif args.scrape_chat:
        exit_code = asyncio.run(run_scrape_chat(args.scrape_chat, scrolls=args.scrolls))
        sys.exit(exit_code)
    else:
        exit_code = asyncio.run(
            run_agent(
                target_username=args.target,
                custom_message=args.message,
                force_available=args.force_available,
                once=args.once,
            )
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

