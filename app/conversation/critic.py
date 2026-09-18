"""Response Quality Critic for Social Conversation (Critic V2).

Detects AI-tells, rhetorical question echo, defensive justifications on banter,
effort parity violations, question budget violations, contextual vagueness,
temporal inconsistencies, handle addressing, and repetitive phrasing.
Produces structured repair constraints for LLM regeneration.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from pydantic import BaseModel, Field

from app.conversation.social_intent import SocialIntent
from app.core.logging import get_logger

logger = get_logger("conversation.critic")


class CriticResult(BaseModel):
    """Evaluation result for a candidate response."""

    passes: bool = True
    score: float = 1.0  # 0.0 to 1.0
    detected_issues: list[str] = Field(default_factory=list)
    repair_constraints: list[str] = Field(default_factory=list)
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
        r"\b(basically|from my perspective|to be honest the|i think the reason is)\b",
        r"\b(great question|that's actually interesting)\b",
        r"\b(maybe you should|perhaps you could|i recommend|i suggest)\b",
    ]

    FORBIDDEN_EMOJIS = {"💀", "🤗", "🤤", "😉", "😊", "🥰", "😜", "😝", "🤪", "😇", "👍"}

    # Insult words that shouldn't be echoed back rhetorically
    ECHO_INSULT_PATTERNS = [
        r"^(dumb\?+|slow\?+|pagal\?+|stupid\?+|gadha\?+|idiot\?+)",
        r"^(toh\?+|toh kya\b|why\?+)",
    ]

    # Defensive justifications when insulted or teased
    DEFENSIVE_JUSTIFICATION_PATTERNS = [
        r"\b(nahi slow speed hai|speed kam hai|actually mai|kisi ne kaha tha|dimag to hai|aisa nahi hai mai|i am not dumb|i'm not stupid)\b",
        r"\b(nahi mai ameer nahi|mai to ordinary student|mai to bas ek|mai koi ameer nahi|aisa nahi hai mai ameer)\b",
        r"\b(pagal nahi hu|kabhi kabhi confuse ho jaati|aisa kuch nahi hai)\b",
        r"\b(mai bata rahi hu|mai bata raha hu|mera matlab ye tha|toh kya matlab)\b",
    ]

    # Low information density / vague non-answers
    VAGUE_NON_ANSWER_PATTERNS = [
        r"\b(kuch nahi bas chill|bas chill kar|kuch nahi bas|bas chill|aise hi chill|timepass kar)\b",
        r"^(bas|thoda|kuch nahi|kuch khas nahi|aise hi|chill|padhai bas|bas padhai|bas chill)[.!?\s]*$",
        r"^(kuch nahi yaar|bas aise hi yaar|bas timepass)[.!?\s]*$",
    ]

    @classmethod
    def evaluate(
        cls,
        candidate_reply: str,
        incoming_text: str,
        intent: SocialIntent,
        strategy: str = "direct_answer",
        state: Any = None,
        delta: Any = None,
        contribution_plan: Any = None,
        recent_replies: Sequence[str] = (),
        target_handle: str | None = None,
    ) -> CriticResult:
        """Evaluate candidate response for conversational quality and social fit."""
        reply_clean = candidate_reply.strip()
        reply_lower = reply_clean.lower()
        incoming_clean = incoming_text.strip()
        incoming_lower = incoming_clean.lower()
        incoming_words = incoming_clean.split()
        reply_words = reply_clean.split()
        in_count = len(incoming_words)
        out_count = len(reply_words)

        issues: list[str] = []
        constraints: list[str] = []
        score = 1.0

        # 1. Check Forbidden Emojis (Hard constraint)
        if "💀" in reply_clean:
            issues.append("forbidden_skull_emoji")
            constraints.append("Do NOT use the skull emoji (💀).")
            score -= 0.5
        found_forbidden = [e for e in cls.FORBIDDEN_EMOJIS if e in reply_clean and e != "💀"]
        if found_forbidden:
            issues.append(f"forbidden_emojis_found:{','.join(found_forbidden)}")
            constraints.append("Do NOT use forbidden emojis (🤗, 😉, 😊, 🥰, 😜, etc.). Use only allowed emojis (😭, 😂) rarely, or use NO emojis.")
            score -= 0.5

        # 2. Check AI-tells and assistant tropes
        for p in cls.AI_TELL_PATTERNS:
            if re.search(p, reply_lower):
                issues.append(f"ai_tell_pattern:{p}")
                constraints.append("Do NOT use assistant language, apologies, textbook explanations, or robotic phrases like 'kyu? kyunki'.")
                score -= 0.45
                break

        # 3. Question budget violations
        has_question_budget_zero = False
        if contribution_plan is not None and getattr(contribution_plan, "question_budget", None) == 0:
            has_question_budget_zero = True
        elif state is not None and getattr(state, "in_interview_mode", False):
            has_question_budget_zero = True

        if has_question_budget_zero and "?" in reply_clean:
            issues.append("question_budget_exceeded")
            constraints.append("Do NOT ask any questions. Question budget is 0. End the message without question marks or inquisitive phrasing.")
            score -= 0.50

        # 4. Contextual vagueness / low information density
        is_activity_or_explanation_query = (
            intent.social_act in ("question_activity", "question_factual")
            or any(w in incoming_lower for w in ["wyd", "kya kar rahi", "kaha hai", "what happened", "kyu", "kyuu"])
            or (contribution_plan is not None and getattr(contribution_plan, "contribution_type", None) in ("direct_answer", "activity_status", "explain_clarify", "answer_probe"))
        )
        if is_activity_or_explanation_query:
            if any(re.search(p, reply_lower) for p in cls.VAGUE_NON_ANSWER_PATTERNS):
                issues.append("contextual_vagueness")
                constraints.append("Be specific and grounded with real details from your routine/life (e.g. physics numericals, school, specific tasks). Do NOT give vague answers like 'bas', 'chill', 'kuch nahi'.")
                score -= 0.45
            elif out_count <= 3 and any(w in reply_lower for w in ["bas", "chill", "kuch nahi"]):
                issues.append("contextual_vagueness")
                constraints.append("Provide a concrete detail about what you are actually doing or thinking.")
                score -= 0.40

        # 5. Temporal inconsistency — asking past tense on ongoing/future events
        ongoing_signals = [
            r"\b(naa?|nahi|no)\s+\d+\s+baje\b",
            r"\b\d+\s+baje\s+(hogi|hoga|khatam)\b",
            r"\b(dekh raha|dekh rahi|khel raha|khel rahi|padh raha|padh rahi|chal raha|chal rahi)\b",
            r"\b(baad me|abhi chal|abhi dekh)\b",
        ]
        past_tense_probes = [
            r"\b(kaisi thi|kaisa tha|khatam ho gay[ai]|khatam ho chuk[ai]|kaisa laga|kaisi lagi|dekh li)\b",
        ]
        is_ongoing_in_text = any(re.search(p, incoming_lower) for p in ongoing_signals)
        is_ongoing_in_state = False
        if state is not None and hasattr(state, "is_subject_ongoing"):
            is_ongoing_in_state = state.is_subject_ongoing("movie") or state.is_subject_ongoing("film")

        if is_ongoing_in_text or is_ongoing_in_state:
            if any(re.search(p, reply_lower) for p in past_tense_probes):
                issues.append("temporal_inconsistency_past_tense_on_ongoing_event")
                constraints.append("The event is ongoing or in the future. Do NOT ask how it was or speak in the past tense. Acknowledge and let them finish.")
                score -= 0.55

        # 6. Inappropriate handle addressing
        handle_to_check = target_handle or (state.contact_id if state and hasattr(state, "contact_id") else None)
        if handle_to_check:
            raw_handle = handle_to_check.lstrip("@").lower()
            if len(raw_handle) >= 3 and raw_handle in reply_lower:
                issues.append("inappropriate_handle_addressing")
                constraints.append(f"Do NOT address the user as '{raw_handle}' or by their Instagram handle. Speak naturally without using their handle.")
                score -= 0.40

        # 7. Banter Mismatch & Defensive Justification
        if intent.social_act in ("playful_insult", "teasing", "rhetorical_challenge") or strategy in ("playful_counter", "continue_joke"):
            if any(re.search(p, reply_lower) for p in cls.ECHO_INSULT_PATTERNS):
                issues.append("rhetorical_insult_echo")
                constraints.append("Do NOT echo the insult back as a rhetorical question (e.g. 'dumb?'). Counter-tease directly.")
                score -= 0.50

            if any(re.search(p, reply_lower) for p in cls.DEFENSIVE_JUSTIFICATION_PATTERNS):
                issues.append("defensive_justification_on_banter")
                constraints.append("Do NOT justify or explain yourself defensively. Counter-tease or roast playfully.")
                score -= 0.50

        # 8. Effort Parity Checks
        if in_count <= 2 and out_count >= 15:
            issues.append(f"effort_parity_overanswering:in_{in_count}_got_{out_count}_words")
            constraints.append("The user sent a 1-2 word message. Match effort parity with a brief reply (under 5 words).")
            score -= 0.45
        elif intent.expected_reply_length == "very_short" and out_count > 8:
            issues.append(f"effort_parity_too_long:expected_very_short_got_{out_count}_words")
            constraints.append("Keep your reply very short (under 5 words).")
            score -= 0.30

        # 9. Unsolicited self-disclosure on short acknowledgment / reaction
        if intent.social_act in ("acknowledgment", "reaction_short", "topic_dismissal") and in_count <= 3:
            self_disclosure_patterns = [
                r"\b(bas thoda|music sun rahi|snack|chill kar rahi|padh rahi|dekh rahi|khati thi|relax kar rahi)\b",
                r"\b(assignments|coaching|physics|school)\b",
            ]
            if out_count > 4 or any(re.search(p, reply_lower) for p in self_disclosure_patterns):
                issues.append("unsolicited_self_disclosure_on_acknowledgment")
                constraints.append("Do NOT share unsolicited updates about yourself or your day. Reply with a simple 1-3 word reaction (e.g. 'haan', 'sahi hai', 'yepp').")
                score -= 0.50

        # 10. Repeated answered question
        if state is not None:
            questions_answered = getattr(state, "questions_answered_by_user", {})
            if questions_answered and "?" in reply_clean:
                for norm_q, answer in questions_answered.items():
                    norm_reply_words = set(re.sub(r"[^\w\s]", "", reply_lower).split())
                    norm_q_words = set(norm_q.split())
                    overlap = norm_reply_words & norm_q_words
                    if len(overlap) >= 2:
                        issues.append(f"repeated_answered_question:{norm_q[:40]}")
                        constraints.append("Do NOT ask a question the user has already answered.")
                        score -= 0.35
                        break

        # 11. Content word repetition against recent character replies
        if recent_replies:
            curr_content_words = set(re.findall(r"[a-zA-Z]{3,}", reply_lower))
            if len(curr_content_words) >= 3:
                for prev in recent_replies[-3:]:
                    if not prev:
                        continue
                    prev_content_words = set(re.findall(r"[a-zA-Z]{3,}", prev.lower()))
                    if len(prev_content_words) >= 3:
                        intersection = curr_content_words & prev_content_words
                        union = curr_content_words | prev_content_words
                        jaccard = len(intersection) / len(union) if union else 0.0
                        if jaccard >= 0.50:
                            issues.append(f"high_content_repetition:jaccard_{jaccard:.2f}")
                            constraints.append("Avoid repeating the same words, subjects, or sentence structure from your recent replies.")
                            score -= 0.45
                            break

        # 12. Unsolicited topic on logistical time query
        if intent.social_act == "question_logistical" or any(w in incoming_lower for w in ["time", "baje", "ghadi", "clock"]):
            unsolicited_topics = [r"\b(film|movie|exam|test|homework|tuition)\b"]
            if any(re.search(p, reply_lower) for p in unsolicited_topics) and "?" in reply_clean:
                issues.append("unsolicited_topic_on_logistical_query")
                constraints.append("Answer the time query directly without bringing up unrelated topics or questions.")
                score -= 0.50

        # 13. Ending Period on casual single-line message
        if "\n" not in reply_clean and reply_clean.endswith(".") and not reply_clean.endswith("..."):
            issues.append("unnatural_ending_period")
            constraints.append("Do NOT end your casual text message with a period.")
            score -= 0.10

        score = max(0.0, round(score, 2))
        has_critical_failure = (
            any("forbidden_emojis_found" in iss for iss in issues)
            or "forbidden_skull_emoji" in issues
            or "rhetorical_insult_echo" in issues
            or "defensive_justification_on_banter" in issues
            or "question_budget_exceeded" in issues
            or "unsolicited_self_disclosure_on_acknowledgment" in issues
            or "temporal_inconsistency_past_tense_on_ongoing_event" in issues
            or "unsolicited_topic_on_logistical_query" in issues
            or "inappropriate_handle_addressing" in issues
            or "contextual_vagueness" in issues
            or any("high_content_repetition" in iss for iss in issues)
            or (in_count <= 2 and out_count >= 15)
            or any("ai_tell_pattern" in iss for iss in issues)
        )

        passes = score >= 0.65 and not has_critical_failure

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
            repair_constraints=constraints,
            suggested_repair=None,
        )
