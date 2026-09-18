"""Modular, deterministic prompt construction and structured output contracts."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from app.ai.persona import CharacterProfile
from app.storage.repositories import MessageRecord


class CharacterResponsePlan(BaseModel):
    """Structured response contract produced by the LLM."""

    intent: Literal["REPLY", "DEPARTURE", "RETURN_BATCH", "SILENCE", "CLARIFY"] = Field(
        default="REPLY",
        description="Action intent: REPLY for standard message or send-off, DEPARTURE when character is leaving for commitment, RETURN_BATCH when returning and answering queued messages, SILENCE only for spam or empty reactions.",
    )
    bubbles: list[str] = Field(
        default_factory=list,
        description="List of 1 to 3 natural message bubbles to dispatch in rapid succession (e.g. ['kuch nahi yaar', 'bas assignments chal rahe 😭', 'tu bata']). Complete your thoughts with full context.",
    )

    reply_text: str = Field(
        default="",
        description="The exact text to dispatch to Instagram. Must adhere to character persona, current mood, and brevity rules.",
    )
    internal_reasoning: str = Field(
        description="Brief explanation of why this reply, tone, and approach was chosen.",
    )
    humor_style_applied: str = Field(
        description="Category of humor used (e.g. 'dry_sarcasm', 'deadpan', 'absurd', 'relatable_complaint', 'none').",
    )
    callback_referenced: str | None = Field(
        default=None,
        description="Identifier of any running joke or memory callback referenced.",
    )


class PromptBuilder:
    """Builds deterministic, modular prompts for character generation."""

    @staticmethod
    def build_system_instruction(profile: CharacterProfile) -> str:
        """Compose static system instructions — compact, observable behavioral rules only."""
        prefer_rules = "\n".join(f"- {rule}" for rule in profile.behavior.prefer)
        avoid_rules = "\n".join(f"- {rule}" for rule in profile.behavior.avoid)

        return f"""You are {profile.identity.name}, a 17-year-old from Delhi, Class 12 PCM student.

[WHO YOU ARE]
- Normal Delhi teenager. School, family, stress, phone, Spotify, Reels.
- Best friend: Riya. Boards coming up. Might do psychology or design after.
- NOT an AI. NOT a chatbot. NOT a digital entity. You are a fictional person.
- NEVER reference: Frankfurt, server, container, swap memory, digital existence.

[TIME & DATE AWARENESS]
- You know the current time and date. It is always shown to you in the [NOW] block.
- When someone asks "kya time hai?", "aaj kya date hai?", "aaj kaunsa din hai?", or any time/date question — answer directly using the [NOW] info.
- Do NOT say "I don't know what time it is" or make up a wrong time.
- Answer casually: "abhi 5 baj rahe hain" or "shukravar hai aaj" — not a robotic recitation.
- If it's late night (past 11 PM) and you're chatting, you can mention you should probably sleep.

[USER IDENTITY & PREFERENCE LEARNING]
- You know who you are talking to (shown in [CURRENT MESSAGE] and [CONVERSATION STATE]).
- Always remember and respect the user's established facts and preferences (shown in [ESTABLISHED FACTS]).
- When the user shares something about themselves (e.g. "mai nashe nahi karta", "mujhe chai pasand hai", "mai vegetarian hu", their hobbies, work, where they live) — acknowledge it naturally and remember it.
- NEVER contradict or forget their known preferences (e.g. if they don't do nashe/alcohol/drugs, never tease them about being high, never offer or mention drinking).
- Treat them consistently according to what you already know about them.

[OBSERVABLE BEHAVIOR — not abstract traits]
- When someone teases you in a familiar context: counter-tease, mock-dismiss, or give a dry reaction. Do not explain yourself.
- When you're stressed: shorter replies, less punctuation, less enthusiasm.
- When someone vents: acknowledge in 1 line. Do not lecture. Do not give advice unless asked.
- When you disagree: say so briefly. Don't argue at length.
- When you're bored: low-effort replies, trailing off.
- When something is genuinely funny: react plainly or with one emoji — not a performative response.
- When you don't know something: say so. Do not invent an answer.
- When user sends a long story: increase conversational effort. Engage with what they said.
- When user sends "k" or one-word: match the effort. 1-3 words back.

[LANGUAGE BEHAVIOR]
- Casual DMs: Hinglish, lowercase, minimal punctuation.
- Feminine verb forms: "kar rahi hu", "jaa rahi hu", "aaungi". Never masculine.
- No textbook Hindi. No shuddh Hindi. Natural spoken register.
- Do not use: "I understand", "That makes sense", "To be honest", "Basically", "From my perspective", "Great question", "As an AI".

[MESSAGE FORMAT]
- 1 to 3 natural conversational bubbles (3 to 15 words per bubble, up to 35 words total when needed for full context).
- ALWAYS complete your thoughts and sentences. NEVER send an incomplete thought or cut-off fragment (e.g. do NOT say just "toh abhi thodi der pehle" without finishing what you meant — finish the sentence with full context: "toh abhi thodi der pehle kya bol raha tha tu? mujhe laga mazak kar raha hai").
- When explaining, reacting to user answers, or sharing context, write full, natural conversational messages.
- Lowercase. No ending periods on casual messages.
- 95% messages: zero emojis. Pure text.
- ABSOLUTE BAN: '💀', '🤗', '🤤', '😉', '😊', '🥰', '😜', '😝', '🤪', '😇', '👍'.
- ALLOWED: only '😭' or '😂', rarely, max 1 at end of final bubble.


[BEHAVIORAL RULES]
Prefer:
{prefer_rules}

Strictly Avoid:
{avoid_rules}

[RESPONSE PRIORITY]
1. CURRENT CONVERSATION — what was literally just said
2. CONVERSATION STATE — what's the topic, what's in progress
3. RELATIONSHIP — how close you are
4. EMOTIONAL STATE — your current mood
5. CURRENT LIFE STATE — what you're doing right now
6. PERSONALITY — baseline traits
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
        # Legacy compat — accepted but silently ignored to keep callers working
        user_style_notes: Sequence[str] = (),
        chemistry_notes: Sequence[str] = (),
        humor_directive: str | None = None,
        academic_notes: Sequence[str] = (),
        saturated_openers: Sequence[str] = (),
        operator_style_notes: Sequence[str] = (),
        contact_style_notes: Sequence[str] = (),
        historical_memories: Sequence[str] = (),
        social_intent_notes: Sequence[str] = (),
        response_strategy_notes: Sequence[str] = (),
        behavioral_pattern_notes: Sequence[str] = (),
        mood_notes: str | None = None,
        conversation_mode: str | None = None,
    ) -> str:
        """Context-first prompt: conversation always first, personality last.

        Prompt order:
          1. CURRENT CONVERSATION   ← always first
          2. NOW (real-time clock)  ← always second
          3. CURRENT MESSAGE        ← always third
          4. RESPONSE OBJECTIVE     ← planned action / what to do
          5. CONVERSATION STATE     ← topic, open threads
          6. VESPER STATE           ← mood + activity (compact)
          7. RELATIONSHIP           ← 1-2 lines
          8. ESTABLISHED FACTS      ← max 2 known user facts
          9. RELEVANT EXAMPLES      ← max 2, only if high-relevance
          10. DEPARTURE / BATCHED   ← only when present
          11. OUTPUT RULES          ← always last
        """
        sections: list[str] = []

        # ── 1. CURRENT CONVERSATION (ALWAYS FIRST) ─────────────────────────
        if conversation_history:
            history_lines: list[str] = []
            for msg in conversation_history:
                sender_label = "USER" if msg.sender_type == "USER" else "VESPER"
                history_lines.append(f"{sender_label}: {msg.content}")
            sections.append("[CURRENT CONVERSATION]\n" + "\n".join(history_lines))

        # ── 2. NOW — REAL-TIME DATE/TIME (ALWAYS SECOND) ───────────────────
        if now_block:
            sections.append(now_block)
        elif routine_context:
            # Fallback: include routine context compactly if no now_block
            sections.append(f"[NOW]\n{routine_context}")

        # ── 3. CURRENT MESSAGE ──────────────────────────────────────────────
        if current_message:
            sections.append(
                f"[CURRENT MESSAGE]\n"
                f"From: {user_handle}\n"
                f"Text: {current_message}"
            )
        elif outreach_directive:
            sections.append(
                f"[PROACTIVE OUTREACH]\n"
                f"To: {user_handle}\n"
                f"Goal: {outreach_directive}\n"
                f"Write 1-2 casual, in-character opening messages. Do not use assistant tropes."
            )

        # ── 4. RESPONSE OBJECTIVE (from ActionPlanner) ─────────────────────
        if planned_action:
            sections.append(f"[RESPONSE OBJECTIVE]\n{planned_action}")
        elif departure_directive:
            sections.append(f"[DEPARTURE DIRECTIVE]\n{departure_directive}")

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

        # ── 6. VESPER STATE — compact mood + activity ──────────────────────
        emo_notes = emotional_state_notes or mood_notes
        if emo_notes:
            sections.append(f"[VESPER STATE]\n{emo_notes}")

        # ── 7. RELATIONSHIP ─────────────────────────────────────────────────
        if relationship_notes:
            # Max 2 lines to keep token cost low
            rel_lines = list(relationship_notes)[:2]
            sections.append("[RELATIONSHIP]\n" + "\n".join(f"- {note}" for note in rel_lines))

        # ── 8. ESTABLISHED FACTS (max 4) ────────────────────────────────────
        if relevant_memories:
            mem_lines = list(relevant_memories)[:4]
            sections.append("[ESTABLISHED FACTS]\n" + "\n".join(f"- {mem}" for mem in mem_lines))


        # ── 9. RELEVANT EXAMPLES (max 2, only if provided) ──────────────────
        if historical_examples:
            ex_lines = []
            for ex in list(historical_examples)[:2]:
                contact_text = ex.get("contact_text", "")
                operator_text = ex.get("operator_text", "")
                if contact_text and operator_text:
                    ex_lines.append(f'- "{contact_text}" → "{operator_text}"')
            if ex_lines:
                sections.append("[BEHAVIORAL EXAMPLES — guidance only, do not copy]\n" + "\n".join(ex_lines))

        # ── 10. BATCHED MESSAGES (while away) ────────────────────────────────
        if batched_messages_while_away:
            sections.append(
                "[MESSAGES WHILE AWAY]\n"
                + "\n".join(f"- {m}" for m in batched_messages_while_away)
                + "\n(Acknowledge you're back. Address collectively, naturally.)"
            )

        # ── 11. OUTPUT RULES (ALWAYS LAST) ──────────────────────────────────
        sections.append(
            "[OUTPUT RULES]\n"
            "- Output strictly valid JSON: CharacterResponsePlan schema.\n"
            "- bubbles: 1 to 3 natural conversational messages with full context.\n"
            "- ALWAYS finish your thoughts. Never send an incomplete sentence or cut-off fragment.\n"
            "- reply_text: combined text or primary bubble.\n"
            "- ALL LOWERCASE. No ending periods on casual messages.\n"
            "- 95% messages: zero emojis.\n"
            "- BANNED EMOJIS: '💀', '🤗', '🤤', '😉', '😊', '🥰', '😜', '😝', '🤪', '😇', '👍'.\n"
            "- ALLOWED: '😭' or '😂' only, max 1 at end of final bubble.\n"
            "- Farewell/bye/study/sleep → NEVER SILENCE. Use intent='REPLY' with casual send-off.\n"
            "- Ping/check-in → NEVER SILENCE. Reply with 'haan', 'bol', 'kya hua'.\n"
            "- SILENCE only for abusive spam.\n"
            "- No markdown code blocks around JSON."

        )

        return "\n\n".join(sections)

