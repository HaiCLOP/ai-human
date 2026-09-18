"""Conversational Replay and Evaluation Harness (CLI).

Replays real and synthetic conversation turns through the Social Conversation Intelligence
pipeline to verify speech-act detection, strategy selection, few-shot retrieval, and critic scoring.

Usage:
  python -m tools.replay_conversations --suite
  python -m tools.replay_conversations --message "Why are you so dumb"
  python -m tools.replay_conversations --message "Why are you so dumb" --live
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.conversation.critic import ResponseQualityCritic
from app.conversation.response_strategy import ResponseStrategySelector
from app.conversation.situation_retriever import SituationAwareRetriever
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.state import ConversationState, update_conversation_state
from app.core.logging import get_logger
from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository

logger = get_logger("tools.replay")

DEFAULT_EVALUATION_SUITE = [
    {
        "name": "Playful Banter Insult (Failure benchmark)",
        "incoming": "Why are you so dumb",
        "expected_act": "playful_insult",
        "expected_strategy": "playful_counter",
        "bad_reply": "dumb? nahi slow speed hai bas tu bhi same level pe ha",
        "good_reply": "tu bhi konsa topper hai waise",
    },
    {
        "name": "Hinglish Laugh Tease",
        "incoming": "tu pagal hai kya 😭",
        "expected_act": ["clarification", "playful_insult"],
        "expected_strategy": ["direct_answer", "playful_counter"],
        "bad_reply": "Actually mai pagal nahi hu bas kabhi kabhi confuse ho jaati hu.",
        "good_reply": "shakal dekh apni pehle 😭",
    },
    {
        "name": "Ultra-Brief Low Effort (Effort parity test)",
        "incoming": "Mast",
        "expected_act": ["reaction_short", "acknowledgment"],
        "expected_strategy": ["direct_answer", "short_acknowledgment"],
        "bad_reply": "Haan bilkul mast hona bhi chahiye! Aur batao tumhara din kaisa raha? Mera to bohot busy tha homework ke sath.",
        "good_reply": "sahi hai",
    },
    {
        "name": "Casual Ping / Greeting",
        "incoming": "kaha hai tu",
        "expected_act": ["question_personal", "greeting"],
        "expected_strategy": "direct_answer",
        "bad_reply": "I am currently at home studying for my examinations. Where are you?",
        "good_reply": "ghar pe hu bol kya scene",
    },
    {
        "name": "Teasing / Flex Accusation",
        "incoming": "bade log ameer ho gaye",
        "expected_act": "teasing",
        "expected_strategy": "playful_counter",
        "bad_reply": "Nahi mai ameer nahi hu, mai to ordinary student hu.",
        "good_reply": "haan Ambani se meeting chal rahi hai",
    },
    {
        "name": "Venting / Bad Exam",
        "incoming": "fml test me hag diya bohot bura",
        "expected_act": "venting",
        "expected_strategy": "supportive_response",
        "bad_reply": "I understand your distress. Next time you should prepare a strict study timetable to ensure optimal results.",
        "good_reply": "arre yaar fml itna kyu padhna hai",
    },
    {
        "name": "Farewell / Sleeping",
        "incoming": "so raha hu bye",
        "expected_act": "farewell",
        "expected_strategy": "short_acknowledgment",
        "bad_reply": "Good night! Wishing you sweet dreams and a refreshing rest.",
        "good_reply": "haan so jaa gn",
    },
]


def evaluate_single_message(
    incoming_text: str,
    db_mgr: Any,
    repo: HistoricalRepository,
    retriever: SituationAwareRetriever,
    mock_reply: str | None = None,
) -> dict[str, Any]:
    """Run a single inbound message through the intelligence pipeline."""
    # 1. Social Intent
    intent = SocialIntentAnalyzer.analyze(incoming_text)

    # 2. Strategy Selection
    strat = ResponseStrategySelector.select_strategy(
        intent=intent,
        conversation_mode="casual",
        incoming_word_count=len(incoming_text.split()),
    )

    # 3. Situation Retriever
    examples = retriever.retrieve_examples(incoming_text, intent, limit=2)

    # 4. Critic Evaluation (if mock_reply provided)
    critic_result = None
    if mock_reply:
        critic_result = ResponseQualityCritic.evaluate(
            candidate_reply=mock_reply,
            incoming_text=incoming_text,
            intent=intent,
            strategy=strat.strategy_name,
        )

    return {
        "incoming": incoming_text,
        "intent": intent,
        "strategy": strat,
        "examples": examples,
        "mock_reply": mock_reply,
        "critic_result": critic_result,
    }


async def run_live_message(incoming_text: str) -> None:
    """Run full live LLM orchestration using active provider."""
    from app.conversation.manager import ConversationManager

    db = get_db_manager()
    mgr = ConversationManager(db=db)
    print(f"\n[LIVE EVALUATION] Processing inbound: '{incoming_text}'")

    reply = await mgr.handle_incoming_message(
        conversation_id="replay_session_001",
        sender_handle="arnav_replay",
        message_text=incoming_text,
        force_available=True,
    )
    print(f"[LIVE CHARACTER REPLY]: {reply}\n")


def run_suite():
    """Run evaluation suite and display scorecards."""
    db = get_db_manager()
    repo = HistoricalRepository(db)
    retriever = SituationAwareRetriever(db, repo=repo)

    passed_count = 0
    total = len(DEFAULT_EVALUATION_SUITE)

    print("=" * 75)
    print("  CONVERSATION INTELLIGENCE V2 — REPLAY & EVALUATION HARNESS")
    print("=" * 75)

    for idx, case in enumerate(DEFAULT_EVALUATION_SUITE, start=1):
        incoming = case["incoming"]
        res = evaluate_single_message(
            incoming_text=incoming,
            db_mgr=db,
            repo=repo,
            retriever=retriever,
            mock_reply=case.get("bad_reply"),
        )

        intent = res["intent"]
        strat = res["strategy"]
        bad_critic = res["critic_result"]

        good_critic = ResponseQualityCritic.evaluate(
            candidate_reply=case["good_reply"],
            incoming_text=incoming,
            intent=intent,
            strategy=strat.strategy_name,
        )

        act_ok = intent.social_act == case["expected_act"] if isinstance(case["expected_act"], str) else intent.social_act in case["expected_act"]
        strat_ok = strat.strategy_name == case["expected_strategy"] if isinstance(case["expected_strategy"], str) else strat.strategy_name in case["expected_strategy"]
        critic_caught_bad = not bad_critic.passes if bad_critic else True
        critic_passed_good = good_critic.passes

        is_passed = act_ok and strat_ok and critic_caught_bad and critic_passed_good
        if is_passed:
            passed_count += 1

        status_tag = "[PASS]" if is_passed else "[FAIL]"
        print(f"\nCase {idx}: {case['name']} -> {status_tag}")
        print(f"  Incoming: \"{incoming}\"")
        print(f"  Detected Act: {intent.social_act} (Expected: {case['expected_act']}) | Playfulness={intent.playfulness:.1f} Hostility={intent.hostility:.1f}")
        print(f"  Selected Strategy: {strat.strategy_name} (Max words: {strat.max_words})")
        print(f"  Tactical Goal: {strat.tactical_prompt[:70]}...")
        if res["examples"]:
            top_ex = res["examples"][0]
            print(f"  Top Historical Example: \"{top_ex['contact_text']}\" -> \"{top_ex['operator_text']}\" (Score: {top_ex['score']})")

        print(f"  Testing Unnatural Reply: \"{case['bad_reply']}\"")
        print(f"    Critic Score: {bad_critic.score} | Passes: {bad_critic.passes} | Issues: {bad_critic.detected_issues}")
        print(f"  Testing Authentic Reply: \"{case['good_reply']}\"")
        print(f"    Critic Score: {good_critic.score} | Passes: {good_critic.passes}")

    print("\n" + "=" * 75)
    print(f"  EVALUATION SUMMARY: {passed_count}/{total} cases passed ({int(passed_count/total*100)}%)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Replay conversations through Social Conversation Intelligence v2.")
    parser.add_argument("--suite", action="store_true", help="Run the full benchmark suite")
    parser.add_argument("--message", type=str, default=None, help="Evaluate a custom incoming message")
    parser.add_argument("--live", action="store_true", help="Run live through LLM provider and full character pipeline")
    args = parser.parse_args()

    if args.message and args.live:
        asyncio.run(run_live_message(args.message))
    elif args.message:
        db = get_db_manager()
        repo = HistoricalRepository(db)
        retriever = SituationAwareRetriever(db, repo=repo)
        res = evaluate_single_message(args.message, db, repo, retriever)
        intent = res["intent"]
        strat = res["strategy"]
        print(f"\nIncoming: \"{args.message}\"")
        print(f"Speech Act: {intent.social_act} (confidence: {intent.confidence:.2f})")
        print(f"Pragmatics: Playfulness={intent.playfulness:.2f}, Seriousness={intent.seriousness:.2f}, Hostility={intent.hostility:.2f}")
        print(f"Brevity Goal: {intent.expected_reply_length}")
        print(f"Strategy: {strat.strategy_name}")
        print(f"Tactical Prompt: {strat.tactical_prompt}")
        print(f"Forbidden: {strat.forbidden_approaches}")
        if res["examples"]:
            print("\nHistorical Few-Shot Examples:")
            for e in res["examples"]:
                print(f"  - \"{e['contact_text']}\" -> \"{e['operator_text']}\" (Score: {e['score']})")
    else:
        run_suite()


if __name__ == "__main__":
    main()
