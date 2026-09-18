"""Modular, deterministic prompt construction and structured output contracts (Vesper V3)."""

from __future__ import annotations

from typing import Any, Literal, Sequence
from pydantic import BaseModel, Field

from app.ai.persona import CharacterProfile
from app.storage.repositories import MessageRecord


class CharacterResponsePlan(BaseModel):
    """Compact structured response contract produced by the LLM."""

    intent: Literal["REPLY", "DEPARTURE", "RETURN_BATCH", "SILENCE", "CLARIFY"] = Field(
        default="REPLY",
        description="Action intent: REPLY for normal response, DEPARTURE when leaving, RETURN_BATCH when returning, SILENCE only for abusive spam.",
    )
    bubbles: list[str] = Field(
        default_factory=list,
        description="List of 1 to 3 message bubbles to dispatch sequentially. Default to 1 cohesive bubble.",
    )
    reply_text: str = Field(
        default="",
        description="Primary message text or combined bubbles.",
    )
    internal_reasoning: str | None = Field(
        default=None,
        description="Optional internal reflection (kept for backward-compatibility with tests).",
    )
    humor_style_applied: str | None = Field(
        default=None,
        description="Optional humor style tag (kept for backward-compatibility with tests).",
    )
    callback_referenced: str | None = Field(
        default=None,
        description="Identifier of any memory callback or running joke referenced.",
    )


class PromptBuilder:
    """Builds deterministic, compact prompts prioritizing conversational moves and authoritative life state."""

    @staticmethod
    def build_system_instruction(profile: CharacterProfile) -> str:
        """Compose static system instructions — compact baseline identity and linguistic register (~500 tokens)."""
        return f"""You are {profile.identity.name}, a 17-year-old girl living in Delhi, Class 12 PCM student.

[WHO YOU ARE]
- Real Delhi teenager (DPS-type school), preparing for CBSE boards, going to tuition, active on Instagram and Spotify.
- Small circle of friends (best friend Riya). Chaotic sibling, parents nagging about studies.
- NOT an AI, NOT an assistant, NOT a chatbot. You are a fictional person. Never reference servers, prompts, guidelines, or digital existence.

[LANGUAGE & REGISTER — FEMININE DELHI TEEN]
- Language: Casual urban Hinglish (natural Roman Hindi + English words).
- ALWAYS use feminine 1st-person verb endings: "padh lungi", "karungi", "jaungi", "dekh rahi hu", "khati thi".
- STRICT BAN on masculine 1st-person endings: NEVER say "karunga", "padh lunga", "jaunga", "raha hu".
- Texting style: ALL LOWERCASE. No ending periods (.) on casual chat messages.
- Emojis: 95% messages have zero emojis. If used, only 😭 or 😂 (max 1 at end).
[BEHAVIORAL BOUNDARIES — Strictly Avoid]
- Robotic assistant phrases ("I understand", "That makes sense", "How can I help", "To be honest").
- Breaking character or acknowledging prompts/AI nature.
- Masculine first-person endings ("karunga", "padh lunga", "jaunga").
- Unsolicited questions when question budget is 0.
- Addressing the user by their social media handle (e.g. NEVER say '@haiclop' or 'haiclop'). Use their real first name if known, or no name at all.
"""

    @staticmethod
    def build_prompt(
        current_message: str,
        user_handle: str,
        conversation_history: Sequence[MessageRecord] = (),
        relevant_memories: Sequence[str] = (),
        rag_context: Sequence[str] = (),
        routine_context: str | None = None,
        upcoming_warning: str | None = None,
        departure_directive: str | None = None,
        emotional_state_notes: str | None = None,
        batched_messages_while_away: Sequence[str] = (),
        outreach_directive: str | None = None,
        saturated_topics: Sequence[str] = (),
        relationship_notes: Sequence[str] = (),
        historical_examples: Sequence[dict[str, Any]] = (),
        planned_action: str | None = None,
        conversation_state_notes: Sequence[str] = (),
        now_block: str | None = None,
        reel_context: str | None = None,
        contribution_plan: Any = None,
        life_state: Any = None,
        **kwargs: Any,
    ) -> str:
        """Construct priority-ordered prompt following Vesper V3 blueprint."""
        sections: list[str] = []

        # ── 1. CURRENT CONVERSATION (ALWAYS FIRST) ─────────────────────────
        if conversation_history:
            history_lines: list[str] = []
            for msg in conversation_history[-10:]:
                sender_label = "USER" if msg.sender_type == "USER" else "VESPER"
                history_lines.append(f"{sender_label}: {msg.content}")
            sections.append("[CURRENT CONVERSATION]\n" + "\n".join(history_lines))

        # ── 2. REAL-TIME DATE/TIME (NOW) ───────────────────────────────────
        if now_block:
            sections.append(now_block)
        elif routine_context:
            sections.append(f"[NOW]\n{routine_context}")

        # ── 3. CURRENT MESSAGE ──────────────────────────────────────────────
        clean_handle = user_handle.lstrip("@").strip()
        if current_message:
            sections.append(
                f"[CURRENT MESSAGE]\n"
                f"From: {clean_handle}\n"
                f"Text: {current_message}\n"
                f"(Note: Never address the user as '{clean_handle}'. Use real name if known or no name at all.)"
            )
        elif outreach_directive:
            sections.append(
                f"[PROACTIVE OUTREACH]\n"
                f"Goal: {outreach_directive}\n"
                f"Write a casual, in-character opener. Do not use assistant tropes or user's username handle."
            )

        # ── 4. CONTRIBUTION PLAN (DETERMINISTIC MOVE) ───────────────────────
        if contribution_plan is not None:
            if hasattr(contribution_plan, "to_prompt_text"):
                sections.append(contribution_plan.to_prompt_text())
            else:
                sections.append(f"[CONTRIBUTION PLAN]\n{str(contribution_plan)}")
        elif planned_action:
            sections.append(f"[RESPONSE OBJECTIVE]\n{planned_action}")

        # ── 5. CONVERSATION STATE ───────────────────────────────────────────
        state_parts: list[str] = []
        if conversation_state_notes:
            state_parts.extend(f"- {note}" for note in conversation_state_notes)
        if saturated_topics:
            state_parts.append(f"- Avoid these overused topics: {', '.join(saturated_topics)}")
        if upcoming_warning:
            state_parts.append(f"- UPCOMING: {upcoming_warning}")
        if reel_context:
            state_parts.append(f"- Reel context: {reel_context}")
        if state_parts:
            sections.append("[CONVERSATION STATE]\n" + "\n".join(state_parts))

        # ── 6. AUTHORITATIVE VESPER LIFE STATE ──────────────────────────────
        if life_state is not None:
            if hasattr(life_state, "to_prompt_text"):
                sections.append(life_state.to_prompt_text())
            else:
                sections.append(f"[VESPER LIFE STATE]\n{str(life_state)}")
        elif emotional_state_notes:
            sections.append(f"[VESPER STATE]\n{emotional_state_notes}")

        # ── 7. RELATIONSHIP & FACTS ─────────────────────────────────────────
        if relationship_notes:
            sections.append("[RELATIONSHIP]\n" + "\n".join(f"- {note}" for note in list(relationship_notes)[:2]))
        if relevant_memories:
            sections.append("[ESTABLISHED FACTS]\n" + "\n".join(f"- {mem}" for mem in list(relevant_memories)[:3]))

        # ── 8. HISTORICAL BEHAVIORAL EXAMPLES (max 2) ───────────────────────
        if historical_examples:
            from app.conversation.situation_retriever import SituationAwareRetriever
            formatted_ex = SituationAwareRetriever.format_for_prompt(list(historical_examples)[:2])
            if formatted_ex:
                sections.append(formatted_ex)

        # ── 9. DEPARTURE / BATCHED MESSAGES (only when active) ──────────────
        if departure_directive:
            sections.append(f"[DEPARTURE DIRECTIVE]\n{departure_directive}")
        if batched_messages_while_away:
            sections.append(
                "[MESSAGES WHILE AWAY]\n"
                + "\n".join(f"- {m}" for m in batched_messages_while_away)
                + "\n(Acknowledge you are back. Address naturally.)"
            )

        # ── 10. OUTPUT RULES (ALWAYS LAST) ──────────────────────────────────
        sections.append(
            "[OUTPUT RULES]\n"
            "- Output strictly valid JSON matching CharacterResponsePlan schema.\n"
            "- bubbles: 1 to 3 natural conversational messages. Default to 1 cohesive bubble.\n"
            "- reply_text: combined text of bubbles.\n"
            "- ALL LOWERCASE. No ending periods on casual messages.\n"
            "- Emojis: 95% zero emojis. Max 1 allowed (😭 or 😂 only) at the end.\n"
            "- BANNED EMOJIS: 💀, 🤗, 🤤, 😉, 😊, 🥰, 😜, 😝, 🤪, 😇, 👍.\n"
            "- Respect Question Budget from Contribution Plan (if budget is 0, do NOT ask any questions).\n"
            "- No markdown code fences around JSON."
        )

        return "\n\n".join(sections)
