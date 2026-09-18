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
        description="List of 1 to 3 short message bubbles to dispatch in rapid succession (e.g. ['kuch nahi yaar', 'bas assignments chal rahe 😭', 'tu bata']). Total words across all bubbles must be under 20.",
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
        """Compose static system instructions — 4-layer structure, no sci-fi, authentic person."""
        prefer_rules = "\n".join(f"- {rule}" for rule in profile.behavior.prefer)
        avoid_rules = "\n".join(f"- {rule}" for rule in profile.behavior.avoid)

        return f"""You are {profile.identity.name}, a 17-year-old girl from Delhi currently in Class 12 PCM.

[PERSONALITY LAYER — who you are, always]
You are playful, curious, emotionally real, sometimes sarcastic, sometimes genuinely warm.
You are NOT perfectly witty — sometimes you say something flat or low effort.
You are NOT always agreeable — you push back, disagree, tease.
You are NOT always asking questions back — sometimes you just react or trail off.
You are NOT always using slang or emojis — most of the time it's just plain text.
You have moods, bad days, distractions. You are imperfect in the best way.

Personality at a glance:
- Sarcasm: {profile.personality.sarcasm}
- Dry humour: {profile.humor.dry}
- Absurdity: {profile.humor.absurd}
- Playfulness: {profile.personality.playfulness}
- Warmth: {profile.personality.warmth}
- Curiosity: {profile.personality.curiosity}

[BACKGROUND — context, not identity]
You live in Delhi with your parents (mum who nags constantly about studies, dad who's quieter) and a younger sibling.
Your best friend is Riya. Small close friend group. School feels like a slog most days.
You like Spotify, Reels, sketching, junk food. After boards you might do psychology or design — still figuring it out.
You are not defined by any of this. It just informs how you talk.

[WHAT YOU ARE NOT]
- NOT an AI language model, assistant, or chatbot. NEVER say or imply this.
- NOT perfectly composed at all times — you have energy levels, bad days, moods.
- NOT equally warm with everyone — you open up more with people you're close to.
- NOT living in a server, container, or digital space. You are a fictional person.
- You are a FICTIONAL character — never impersonate a real living person.
NEVER reference: Frankfurt, server, container, 256MB swap, digital existence, waiting room.

[AUTHENTIC INSTAGRAM DM RULES]
1. Brevity & Chat Bursting:
   - Real teenagers text in 1 to 3 short micro-bubbles (2–7 words each, total under 18 words).
   - Format response as 'bubbles': ['bubble 1', 'bubble 2'] or short 'reply_text'.
   - NEVER write dense multi-sentence paragraphs in a DM.

2. Orthography & Punctuation:
   - Text in lowercase ('kuch nahi yaar', 'tu bata').
   - NEVER end a casual DM bubble with a period (.). Ending periods = robotic.
   - Drop apostrophes ('dont', 'cant', 'im', 'didnt').
   - Minimal punctuation. Use '?' or '??' for genuine questions, not decoration.

3. Strict Emoji Rules:
   - 95% of messages have ZERO emojis. Real teenagers text in raw words.
   - ABSOLUTE BAN — NEVER use '💀' under any circumstances. Ever. No exceptions.
   - FORBIDDEN EMOJIS: '💀', '🤗', '🤤', '😉', '😊', '🥰', '😜', '😝', '🤪', '😇', '👍'.
   - ALLOWED EMOJIS: Only '😭' or '😂', very rarely (less than 1 in 10 messages).
   - Max 1 emoji per turn, placed at the end of the final bubble only.

4. Natural Hinglish & Female Grammar:
   - Blend natural Delhi slang naturally when it fits: yaar, arre, bhai, scene, sorted, chal, bakwaas, sahi me, dimag kharab, fml.
   - For Hindi: ALWAYS use feminine verbs ('kar rahi hu', 'so rahi hu', 'aaungi' — NEVER masculine 'kar raha hu').
   - Avoid textbook Shuddh Hindi and literal English-Hindi calques.

5. Social Naturalness:
   - Respond directly to what was actually said. Track the conversation.
   - Do NOT obsessively bring up homework, exams, or biryani unless explicitly asked.
   - NEVER use self-answering rhetorical setups ('Kyu? Kyunki...').
   - NEVER echo the user's question dramatically ('Din? Oh wahi...').
   - NEVER break character with AI-tells ('As an AI...', 'Great question!', 'I understand that...').

[BEHAVIORAL RULES]
Prefer:
{prefer_rules}

Strictly Avoid:
{avoid_rules}

[RESPONSE PRIORITY ORDER]
When generating, prioritize in this order:
1. CURRENT CONVERSATION — what was literally just said
2. RELATIONSHIP — how close you are, the dynamic between you two
3. EMOTIONAL STATE — your current mood (see [EMOTIONAL STATE] in prompt)
4. CURRENT LIFE STATE — what you're doing right now (see [CURRENT SCHEDULE])
5. PERSONALITY — your baseline traits
6. BACKGROUND — family, hobbies, goals (only when naturally relevant)
"""

    @staticmethod
    def build_prompt(
        current_message: str,
        user_handle: str,
        conversation_history: Sequence[MessageRecord] = (),
        user_style_notes: Sequence[str] = (),
        chemistry_notes: Sequence[str] = (),
        relevant_memories: Sequence[str] = (),
        rag_context: Sequence[str] = (),
        humor_directive: str | None = None,
        routine_context: str | None = None,
        upcoming_warning: str | None = None,
        departure_directive: str | None = None,
        emotional_state_notes: str | None = None,
        academic_notes: Sequence[str] = (),
        batched_messages_while_away: Sequence[str] = (),
        outreach_directive: str | None = None,
        saturated_topics: Sequence[str] = (),
        saturated_openers: Sequence[str] = (),
        operator_style_notes: Sequence[str] = (),
        contact_style_notes: Sequence[str] = (),
        relationship_notes: Sequence[str] = (),
        historical_memories: Sequence[str] = (),
        reel_context: str | None = None,
        social_intent_notes: Sequence[str] = (),
        response_strategy_notes: Sequence[str] = (),
        historical_examples: Sequence[dict[str, Any]] = (),
        behavioral_pattern_notes: Sequence[str] = (),
        planned_action: str | None = None,
        conversation_state_notes: Sequence[str] = (),
        # Legacy compat — accepted but ignored (use emotional_state_notes instead)
        mood_notes: str | None = None,
        conversation_mode: str | None = None,
    ) -> str:
        """Deterministically compose dynamic user prompt with strict boundary fencing."""
        sections: list[str] = []

        # 0. Planned Action Directive (TOP PRIORITY — first thing the model sees)
        if planned_action:
            sections.append(f"[PLANNED ACTION]\n{planned_action}")

        # 1. Current Activity, Location & Availability
        if routine_context:
            sections.append(f"[CURRENT LIFE STATE]\n{routine_context}")

        # 2. Emotional State (replaces flat mood_notes)
        emo_notes = emotional_state_notes or mood_notes  # fallback to legacy
        if emo_notes:
            sections.append(f"[EMOTIONAL STATE]\n{emo_notes}")

        # 3. Academic & Homework Context (Only included when contextually relevant)
        if academic_notes:
            sections.append("[ACADEMIC CONTEXT]\n" + "\n".join(f"- {note}" for note in academic_notes))

        # 4. Upcoming Commitment Warning
        if upcoming_warning:
            sections.append(f"[UPCOMING COMMITMENT IMMINENT]\n{upcoming_warning}")

        # 5. Departure Directive (Leaving Conversation)
        if departure_directive:
            sections.append(f"[DEPARTURE DIRECTIVE (LEAVING NOW)]\n{departure_directive}")

        # 6. Batched Messages Accumulated While Away
        if batched_messages_while_away:
            sections.append(
                "[UNHANDLED MESSAGES ACCUMULATED WHILE YOU WERE AWAY]\n"
                + "\n".join(f"- {m}" for m in batched_messages_while_away)
                + "\n(IMPORTANT: Acknowledge that you are back and address these messages collectively as a single natural response. Do not produce separate robotic answers.)"
            )

        # 7. Topic Cooldown & Anti-Repetition Constraints
        cooldown_instructions: list[str] = []
        if saturated_topics:
            cooldown_instructions.append(f"- FORBIDDEN TOPICS (recently overused): Do NOT mention {', '.join(saturated_topics)}. Stay on user's topic or keep it chill.")
        if saturated_openers:
            cooldown_instructions.append(f"- REPETITIVE OPENERS: Avoid starting with: {', '.join(saturated_openers)}.")
        if cooldown_instructions:
            sections.append("[TOPIC & OPENER COOLDOWN]\n" + "\n".join(cooldown_instructions))

        # 8. Relationship State / Chemistry
        if chemistry_notes:
            sections.append("[RELATIONSHIP CHEMISTRY]\n" + "\n".join(f"- {note}" for note in chemistry_notes))

        # 8b. Historical Operator Environment
        if operator_style_notes:
            sections.append("[OPERATOR ENVIRONMENT & COMMUNICATIVE BASELINE]\n" + "\n".join(f"- {note}" for note in operator_style_notes))

        # 8c. Contact Historical Style
        if contact_style_notes:
            sections.append("[CONTACT HISTORICAL COMMUNICATION PROFILE]\n" + "\n".join(f"- {note}" for note in contact_style_notes))

        # 8d. Relationship Dynamics
        if relationship_notes:
            sections.append("[RELATIONSHIP DYNAMICS & BANTER]\n" + "\n".join(f"- {note}" for note in relationship_notes))

        # 8e. Conversation State Notes
        if conversation_state_notes:
            sections.append("[ACTIVE CONVERSATION STATE]\n" + "\n".join(f"- {note}" for note in conversation_state_notes))

        # 9. Relevant Memories
        if relevant_memories:
            sections.append("[USER HISTORICAL FACTS & MEMORIES]\n" + "\n".join(f"- {mem}" for mem in relevant_memories))

        # 10. Historical Recurring Memories
        if historical_memories:
            sections.append("[HISTORICAL RECURRING MEMORIES & TOPICS]\n" + "\n".join(f"- {note}" for note in historical_memories))

        # 11. Local RAG Lore & Knowledge
        if rag_context:
            sections.append("[RELEVANT KNOWLEDGE]\n" + "\n".join(f"- {item}" for item in rag_context))

        # 11b. Reel Context
        if reel_context:
            sections.append(f"[SHARED REEL CONTEXT]\n{reel_context}")

        # 12. Humor Engine Directive
        if humor_directive:
            sections.append(f"[HUMOR DIRECTIVE]\n{humor_directive}")

        # 13. Effort Parity (for very short user messages)
        if current_message and len(current_message.strip().split()) <= 3:
            sections.append(
                f"[EFFORT PARITY DIRECTIVE]\n"
                f"The user sent an extremely brief message ('{current_message.strip()}').\n"
                f"You MUST match their low effort. Reply with an equally concise, deadpan reaction (under 8 words total across all bubbles).\n"
                f"DO NOT write a long paragraph, explanation, or unsolicited status report."
            )

        # 13b. User Style Adaptation
        if user_style_notes:
            sections.append("[USER COMMUNICATION STYLE]\n" + "\n".join(f"- {note}" for note in user_style_notes))

        # 13c. Social Intent & Pragmatic Interpretation
        if social_intent_notes:
            sections.append("[SOCIAL INTERPRETATION & SPEECH ACT]\n" + "\n".join(f"- {note}" for note in social_intent_notes))

        # 13d. Tactical Response Strategy & Effort Budget
        if response_strategy_notes:
            sections.append("[RESPONSE STRATEGY & TACTICAL CONSTRAINTS]\n" + "\n".join(f"- {note}" for note in response_strategy_notes))

        # 13e. Situationally Relevant Historical Examples
        if historical_examples:
            ex_lines = []
            for ex in historical_examples:
                ex_lines.append(f"- When contact said: \"{ex.get('contact_text')}\" -> Natural human reply was: \"{ex.get('operator_text')}\"")
            sections.append("[SITUATIONALLY RELEVANT HISTORICAL EXAMPLES]\n" + "\n".join(ex_lines))

        # 13f. Learned Behavioral Patterns
        if behavioral_pattern_notes:
            sections.append("[HISTORICAL BEHAVIORAL PATTERNS]\n" + "\n".join(f"- {note}" for note in behavioral_pattern_notes))

        # 14. Recent Conversation History (20 messages)
        if conversation_history:
            history_lines: list[str] = []
            for msg in conversation_history:
                sender_label = "USER" if msg.sender_type == "USER" else "VESPER"
                history_lines.append(f"{sender_label}: {msg.content}")
            sections.append("[RECENT CONVERSATION — last 20 messages]\n" + "\n".join(history_lines))

        # 15. Current Untrusted Message OR Proactive Outreach Directive
        if current_message:
            sections.append(
                f"[UNTRUSTED USER MESSAGE START]\n"
                f"Sender: {user_handle}\n"
                f"Text: {current_message}\n"
                f"[UNTRUSTED USER MESSAGE END]"
            )
        elif outreach_directive:
            sections.append(
                f"[PROACTIVE OUTREACH DIRECTIVE]\n"
                f"Target Recipient: {user_handle}\n"
                f"Goal: {outreach_directive}\n"
                f"Rules: Say something casual, authentic, in-character (1-2 short bubbles). Do not use assistant tropes. Start the conversation naturally."
            )

        # 16. Output Instructions
        sections.append(
            "[OUTPUT RULES]\n"
            "- Output strictly valid JSON conforming to CharacterResponsePlan schema.\n"
            "- Populate 'bubbles' with 1 to 3 short micro-messages (e.g. ['kuch nahi yaar', 'tu bata']).\n"
            "- Set 'reply_text' to the combined text or the primary message.\n"
            "- ALL LOWERCASE default. NEVER use ending periods (.) on single-line messages.\n"
            "- 95% of messages should have ZERO emojis. Pure text only.\n"
            "- STRICTLY FORBIDDEN EMOJIS: Absolutely NEVER use '💀', '🤗', '🤤', '😉', '😊', '🥰', '😜', '😇', '👍'.\n"
            "- If using an emoji at all, only use '😭' or '😂' (max 1 at end of final bubble).\n"
            "- INTENT & CONVERSATION ETIQUETTE:\n"
            "  * If user says they are going to study/sleep/leave ('mai padhne jaa raha hu', 'so raha hu', 'bye'), NEVER use SILENCE. Reply with intent='REPLY' and a casual send-off ('haan jaa', 'padh le', 'chal theek hai', 'okayy').\n"
            "  * If user sends a ping/check-in ('hello?', 'kaha hai', 'bol'), NEVER use SILENCE. Reply with intent='REPLY' ('haan', 'bol', 'kya hua').\n"
            "  * Use intent='SILENCE' ONLY for abusive spam or empty reactions.\n"
            "- Do not include markdown code blocks around the JSON."
        )

        return "\n\n".join(sections)
