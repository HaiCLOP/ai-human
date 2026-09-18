"""Response Quality Critic for Social Conversation.

Detects AI-tells, rhetorical question echo, defensive justifications on banter,
effort parity violations, and unnatural formatting.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from app.conversation.social_intent import SocialIntent
from app.core.logging import get_logger

logger = get_logger("conversation.critic")


class CriticResult(BaseModel):
    """Evaluation result for a candidate response."""

    passes: bool = True
    score: float = 1.0  # 0.0 to 1.0
    detected_issues: list[str] = Field(default_factory=list)
    suggested_repair: str | None = None


class ResponseQualityCritic:
    """Evaluates candidate character replies against social intent and natural conversation rules."""

    AI_TELL_PATTERNS = [
        r"\b(kyu\? kyunki|kyun\? kyunki|why\? because)\b",
        r"\b(as an ai|language model|i am a bot|virtual assistant)\b",
        r"\b(i understand (that|how you feel|your|what)|that makes (total )?sense|i apologize for)\b",
        r"\b(let me explain|actually i am|to be honest the reason)\b",
        r"\b(haha that's funny|great to hear that|how can i help|how may i help)\b",
        r"\b(din kaisa raha|how was your day|tumhara din kaisa)\b",
        r"\b(currently at home studying for my examinations|as per my schedule|at home studying for my exams)\b",
        r"\b(wishing you (sweet|a|rest)|sweet dreams|refreshing rest|safe travels)\b",
        r"\b(next time you should|to ensure optimal|prepare a strict study timetable)\b",
    ]

    FORBIDDEN_EMOJIS = {"💀", "🤗", "🤤", "😉", "😊", "🥰", "😜", "😝", "🤪", "😇", "👍"}

    # Insult words that shouldn't be echoed back rhetorically
    ECHO_INSULT_PATTERNS = [
        r"^(dumb\?+|slow\?+|pagal\?+|stupid\?+|gadha\?+|idiot\?+)",
    ]

    # Defensive justifications when insulted or teased
    DEFENSIVE_JUSTIFICATION_PATTERNS = [
        r"\b(nahi slow speed hai|speed kam hai|actually mai|kisi ne kaha tha|dimag to hai|aisa nahi hai mai|i am not dumb|i'm not stupid)\b",
        r"\b(nahi mai ameer nahi|mai to ordinary student|mai to bas ek|mai koi ameer nahi|aisa nahi hai mai ameer)\b",
    ]

    @classmethod
    def evaluate(
        cls,
        candidate_reply: str,
        incoming_text: str,
        intent: SocialIntent,
        strategy: str = "direct_answer",
    ) -> CriticResult:
        """Evaluate candidate response for conversational quality and social fit."""
        reply_clean = candidate_reply.strip()
        reply_lower = reply_clean.lower()
        incoming_words = incoming_text.strip().split()
        reply_words = reply_clean.split()
        issues: list[str] = []
        score = 1.0

        # 1. Check Forbidden Emojis (Hard constraint)
        if "💀" in reply_clean:
            issues.append("forbidden_skull_emoji")
            score -= 0.5
        found_forbidden = [e for e in cls.FORBIDDEN_EMOJIS if e in reply_clean]
        if found_forbidden:
            issues.append(f"forbidden_emojis_found:{','.join(found_forbidden)}")
            score -= 0.3

        # 2. Check AI-tells and assistant tropes
        for p in cls.AI_TELL_PATTERNS:
            if re.search(p, reply_lower):
                issues.append(f"ai_tell_pattern:{p}")
                score -= 0.45

        # 3. Banter Mismatch & Defensive Justification
        if intent.social_act in ("playful_insult", "teasing") or strategy in ("playful_counter", "continue_joke"):
            # Echoing the insult e.g. "dumb? nahi..."
            if any(re.search(p, reply_lower) for p in cls.ECHO_INSULT_PATTERNS):
                issues.append("rhetorical_insult_echo")
                score -= 0.5

            # Defensive justification
            if any(re.search(p, reply_lower) for p in cls.DEFENSIVE_JUSTIFICATION_PATTERNS):
                issues.append("defensive_justification_on_banter")
                score -= 0.5

        # 4. Effort Parity Checks
        in_count = len(incoming_words)
        out_count = len(reply_words)

        if in_count <= 2 and out_count >= 15:
            issues.append(f"effort_parity_overanswering:in_{in_count}_got_{out_count}_words")
            score -= 0.5
        elif intent.expected_reply_length == "very_short" and out_count > 12:
            issues.append(f"effort_parity_too_long:expected_very_short_got_{out_count}_words")
            score -= 0.35
        elif in_count <= 4 and out_count > 18:
            issues.append(f"effort_parity_overanswering:in_{in_count}_got_{out_count}_words")
            score -= 0.4

        # 5. Ending Period on casual single-line message
        if "\n" not in reply_clean and reply_clean.endswith(".") and not reply_clean.endswith("..."):
            issues.append("unnatural_ending_period")
            score -= 0.1

        score = max(0.0, round(score, 2))
        has_critical_failure = (
            "forbidden_skull_emoji" in issues
            or "rhetorical_insult_echo" in issues
            or "defensive_justification_on_banter" in issues
            or any("effort_parity_overanswering" in iss for iss in issues)
            or any("ai_tell_pattern" in iss for iss in issues)
        )
        passes = score >= 0.6 and not has_critical_failure

        # Construct suggested repair if failed
        suggested_repair = None
        if not passes:
            if "rhetorical_insult_echo" in issues or "defensive_justification_on_banter" in issues:
                suggested_repair = "tu bhi kam nahi hai waise"
            elif "forbidden_skull_emoji" in issues:
                clean_no_skull = reply_clean.replace("💀", "")
                suggested_repair = clean_no_skull.strip()

        logger.debug(
            "critic.evaluated",
            passes=passes,
            score=score,
            issues_count=len(issues),
        )

        return CriticResult(
            passes=passes,
            score=score,
            detected_issues=issues,
            suggested_repair=suggested_repair,
        )
