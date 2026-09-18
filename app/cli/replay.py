"""Conversational Replay and Evaluation CLI for Vesper Intelligence V3.

Replays real conversation threads or benchmark suites through the
Social Intent -> Policy -> Contribution Plan -> Critic V2 pipeline.

Usage:
  python -m app.cli replay --suite
  python -m app.cli replay --conversation <conversation_id>
  python -m app.cli replay --message "Mast"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

# Ensure UTF-8 output in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.conversation.critic import ResponseQualityCritic
from app.conversation.policy import ConversationPolicy
from app.conversation.response_strategy import ResponseStrategySelector
from app.conversation.situation_retriever import SituationAwareRetriever
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.state import ConversationState, update_conversation_state
from app.core.logging import get_logger
from app.routine.manager import RoutineManager, to_ist
from app.routine.models import VesperLifeState
from app.routine.mood import MoodEngine
from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository
from app.storage.repositories import ConversationRepository, MessageRepository

logger = get_logger("cli.replay")

BENCHMARK_SUITE = [
    {
        "name": "Playful Banter Insult (Failure benchmark)",
        "incoming": "Why are you so dumb",
        "expected_act": "playful_insult",
        "bad_reply": "dumb? nahi slow speed hai bas tu bhi same level pe ha",
        "good_reply": "tu bhi konsa topper hai waise",
    },
    {
        "name": "Hinglish Laugh Tease",
        "incoming": "tu pagal hai kya 😭",
        "expected_act": "playful_insult",
        "bad_reply": "Actually mai pagal nahi hu bas kabhi kabhi confuse ho jaati hu.",
        "good_reply": "shakal dekh apni pehle 😭",
    },
    {
        "name": "Ultra-Brief Low Effort (Effort parity test)",
        "incoming": "Mast hai",
        "expected_act": "reaction_short",
        "bad_reply": "Haan bilkul mast hona bhi chahiye! Aur batao tumhara din kaisa raha? Mera to bohot busy tha homework ke sath.",
        "good_reply": "sahi hai na",
    },
    {
        "name": "Ongoing Event (No premature past-tense probe)",
        "incoming": "Naa 7 baje hogi",
        "expected_act": "ongoing_activity",
        "bad_reply": "Achha kaisa tha movie? Maza aaya?",
        "good_reply": "achha theek hai fir",
    },
    {
        "name": "Activity Query with Grounded Life State",
        "incoming": "wyd",
        "expected_act": "activity_query",
        "bad_reply": "kuch nahi bas chill kar rahi hu",
        "good_reply": "physics ke numericals dekh rahi hu dimag kharab ho raha",
    },
    {
        "name": "Why Probe expecting explanation",
        "incoming": "kyu?",
        "expected_act": "clarification",
        "bad_reply": "bas padhai",
        "good_reply": "kal test hai aur kuch aata nahi 😭",
    },
    {
        "name": "Topic Dismissal / Closer",
        "incoming": "chhoro",
        "expected_act": "topic_dismissal",
        "bad_reply": "nahi batao na kya hua please",
        "good_reply": "haan chhoro",
    },
    {
        "name": "Rhetorical Challenge",
        "incoming": "Toh?",
        "expected_act": "rhetorical_challenge",
        "bad_reply": "toh kya matlab mai bata rahi hu tumko",
        "good_reply": "kuch nahi re",
    },
]


def evaluate_benchmark_suite():
    """Run the 8 standard Vesper conversational intelligence benchmark cases."""
    print("=" * 75)
    print("  VESPER CONVERSATIONAL INTELLIGENCE V3 — BENCHMARK EVALUATION")
    print("=" * 75)

    passed_count = 0
    total = len(BENCHMARK_SUITE)

    for idx, case in enumerate(BENCHMARK_SUITE, start=1):
        incoming = case["incoming"]
        intent = SocialIntentAnalyzer.analyze(incoming)
        plan = ConversationPolicy.evaluate(
            clean_text=incoming,
            social_intent=intent,
            state=ConversationState(conversation_id="test", contact_id="user"),
            life_state=VesperLifeState(
                activity="Studying Physics",
                location="study desk",
                current_task="solving numericals",
                energy=0.5,
                mood_label="exhausted",
                free_time=False,
                next_event="dinner at 21:00",
                unfinished_thought="why is thermodynamics so hard",
            ),
            rel_style={"playfulness": 0.7, "response_style": "casual"},
            mood=None,
        )

        bad_critic = ResponseQualityCritic.evaluate(
            candidate_reply=case["bad_reply"],
            incoming_text=incoming,
            intent=intent,
            contribution_plan=plan,
        )

        good_critic = ResponseQualityCritic.evaluate(
            candidate_reply=case["good_reply"],
            incoming_text=incoming,
            intent=intent,
            contribution_plan=plan,
        )

        act_ok = intent.social_act == case["expected_act"] or case["expected_act"] in intent.social_act
        bad_rejected = not bad_critic.passes
        good_passed = good_critic.passes

        case_passed = bad_rejected and good_passed
        if case_passed:
            passed_count += 1

        status = "[PASS]" if case_passed else "[FAIL]"
        print(f"\nCase {idx}: {case['name']} -> {status}")
        print(f"  Incoming: \"{incoming}\"")
        print(f"  Detected Act: {intent.social_act} | Question Budget: {plan.question_budget}")
        print(f"  Contribution Type: {plan.contribution_type.value} | Desired Length: {plan.desired_length}")
        print(f"  Testing Unnatural Reply: \"{case['bad_reply']}\"")
        print(f"    Critic Score: {bad_critic.score} | Passes: {bad_critic.passes} | Issues: {bad_critic.detected_issues}")
        print(f"  Testing Authentic Reply: \"{case['good_reply']}\"")
        print(f"    Critic Score: {good_critic.score} | Passes: {good_critic.passes}")

    print("\n" + "=" * 75)
    print(f"  EVALUATION SUMMARY: {passed_count}/{total} cases passed ({int(passed_count/total*100)}%)")
    print("=" * 75 + "\n")


def replay_conversation(conversation_id: str):
    """Replay an existing conversation from the database and evaluate conversational metrics."""
    db = get_db_manager()
    msg_repo = MessageRepository(db)
    messages = msg_repo.get_recent_messages(conversation_id, limit=50)

    if not messages:
        print(f"No messages found for conversation '{conversation_id}'.")
        return

    print("=" * 75)
    print(f"  REPLAYING CONVERSATION: {conversation_id} ({len(messages)} messages)")
    print("=" * 75)

    total_turns = 0
    critic_passed_turns = 0
    questions_asked = 0
    total_words = 0

    state = ConversationState(conversation_id=conversation_id)

    for i, m in enumerate(messages):
        if m.sender_type == "USER":
            incoming_text = m.content
            intent = SocialIntentAnalyzer.analyze(incoming_text)
            plan = ConversationPolicy.evaluate(
                clean_text=incoming_text,
                social_intent=intent,
                state=state,
            )

            # Look ahead for character reply if available
            char_reply = None
            if i + 1 < len(messages) and messages[i + 1].sender_type == "CHARACTER":
                char_reply = messages[i + 1].content

            print(f"\n[Turn {total_turns + 1}] User: \"{incoming_text}\"")
            print(f"  Intent: {intent.social_act} | Plan: {plan.contribution_type.value} (Budget: {plan.question_budget} questions)")

            if char_reply:
                critic_res = ResponseQualityCritic.evaluate(
                    candidate_reply=char_reply,
                    incoming_text=incoming_text,
                    intent=intent,
                    contribution_plan=plan,
                    state=state,
                )
                total_turns += 1
                if critic_res.passes:
                    critic_passed_turns += 1
                if "?" in char_reply:
                    questions_asked += 1
                total_words += len(char_reply.split())

                pass_str = "PASS" if critic_res.passes else "FLAGGED"
                print(f"  Character: \"{char_reply}\"")
                print(f"  Critic: [{pass_str}] Score: {critic_res.score} | Issues: {critic_res.detected_issues}")

                state = update_conversation_state(
                    state,
                    intent,
                    strategy_name=plan.contribution_type.value,
                    vesper_reply=char_reply,
                    current_user_text=incoming_text,
                )

    print("\n" + "=" * 75)
    print(f"  CONVERSATION REPLAY METRICS:")
    print(f"  Total Character Turns: {total_turns}")
    if total_turns > 0:
        print(f"  Critic Pass Rate: {int(critic_passed_turns / total_turns * 100)}% ({critic_passed_turns}/{total_turns})")
        print(f"  Questions Asked: {questions_asked} ({int(questions_asked / total_turns * 100)}% of turns)")
        print(f"  Average Reply Length: {round(total_words / total_turns, 1)} words")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Vesper Conversation Intelligence Replay CLI")
    parser.add_argument("command", nargs="?", default="replay", help="Command to run (replay)")
    parser.add_argument("--suite", action="store_true", help="Run standard benchmark suite")
    parser.add_argument("--conversation", type=str, default=None, help="Conversation ID to replay")
    parser.add_argument("--message", type=str, default=None, help="Evaluate a single message")
    args = parser.parse_args()

    if args.conversation:
        replay_conversation(args.conversation)
    elif args.message:
        intent = SocialIntentAnalyzer.analyze(args.message)
        plan = ConversationPolicy.evaluate(
            clean_text=args.message,
            social_intent=intent,
        )
        print(f"Message: {args.message}")
        print(f"Social Act: {intent.social_act}")
        print(f"Contribution Plan: {plan.contribution_type.value}")
        print(f"Question Budget: {plan.question_budget}")
        print(f"Desired Length: {plan.desired_length}")
        print(f"Directive: {plan.reason}")
    else:
        evaluate_benchmark_suite()


if __name__ == "__main__":
    main()
