"""Social Response Strategy Selector.

Selects situational response strategies and produces tactical directives
for LLM generation and effort parity enforcement.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from app.conversation.social_intent import SocialIntent
from app.core.logging import get_logger

logger = get_logger("conversation.response_strategy")


class ResponseStrategy(BaseModel):
    """Execution strategy guiding prompt construction and candidate validation."""

    strategy_name: str
    tactical_prompt: str
    max_words: int = 15
    max_bubbles: int = 2
    forbidden_approaches: list[str] = Field(default_factory=list)
    effort_parity_directive: str = ""


class ResponseStrategySelector:
    """Selects tactical conversational strategies based on intent, relationship, and history."""

    @classmethod
    def select_strategy(
        cls,
        intent: SocialIntent,
        relationship_profile: dict[str, Any] | None = None,
        conversation_mode: str = "casual",
        behavioral_pattern: dict[str, Any] | None = None,
        incoming_word_count: int = 5,
    ) -> ResponseStrategy:
        """Select optimal response strategy and tactical constraints."""
        act = intent.social_act
        rel_playfulness = relationship_profile.get("playfulness", 0.5) if relationship_profile else 0.5

        # Check if a learned behavioral pattern provides a strong strategy
        if behavioral_pattern and behavioral_pattern.get("confidence", 0.0) >= 0.65:
            pat_strat = behavioral_pattern.get("response_strategy")
            if pat_strat:
                logger.debug("strategy.using_historical_pattern", strategy=pat_strat, act=act)
                return cls._build_strategy_object(pat_strat, intent, incoming_word_count)

        # Rule-based strategy resolution
        if act == "playful_insult":
            # Banter counter or playful deflection
            return ResponseStrategy(
                strategy_name="playful_counter",
                tactical_prompt=(
                    "The other person is playfully teasing/insulting you. "
                    "Counter with a quick witty tease, deadpan sarcasm, or an absurd comeback under 8 words. "
                    "DO NOT explain yourself. DO NOT say 'dumb?' or 'nahi'. Treat it as effortless banter."
                ),
                max_words=8,
                max_bubbles=2,
                forbidden_approaches=[
                    "literal explanation",
                    "apologizing",
                    "asking why they said that",
                    "echoing 'dumb?' or 'stupid?'",
                    "defensive justifications",
                ],
                effort_parity_directive="Match their short tease with an equally short or shorter counter (under 8 words total).",
            )

        if act == "teasing":
            return ResponseStrategy(
                strategy_name="playful_counter",
                tactical_prompt=(
                    "The user is teasing you. Tease them back playfully with dry wit or dramatic sarcasm under 10 words. "
                    "Keep it light, authentic Delhi texting tone."
                ),
                max_words=10,
                max_bubbles=2,
                forbidden_approaches=["defensiveness", "taking it seriously", "long explanations"],
                effort_parity_directive="Keep it brief and snappy under 10 words.",
            )

        if act == "reaction_laugh":
            return ResponseStrategy(
                strategy_name="short_acknowledgment",
                tactical_prompt="The user is laughing. Give a deadpan reaction or single witty remark (1-5 words).",
                max_words=6,
                max_bubbles=1,
                forbidden_approaches=["asking what's funny", "long paragraphs", "multiple questions"],
                effort_parity_directive="User just laughed. Send at most 1-4 words.",
            )

        if act == "greeting":
            return ResponseStrategy(
                strategy_name="direct_answer",
                tactical_prompt="Casual greeting back. Reply naturally in 1-2 short bubbles ('haan bol', 'yo kya scene', 'kuch nahi tu bata').",
                max_words=10,
                max_bubbles=2,
                forbidden_approaches=["corporate greetings", "'how was your day'", "long status reports"],
                effort_parity_directive="Match low-effort greeting with casual 2-6 words.",
            )

        if act == "farewell":
            return ResponseStrategy(
                strategy_name="short_acknowledgment",
                tactical_prompt="Say a quick casual goodbye or acknowledge they are sleeping/leaving ('haan jaa', 'gn', 'padh le').",
                max_words=6,
                max_bubbles=1,
                forbidden_approaches=["holding them back", "asking questions", "long goodbyes"],
                effort_parity_directive="Ultra-brief sendoff under 5 words.",
            )

        if act == "acknowledgment":
            return ResponseStrategy(
                strategy_name="short_acknowledgment",
                tactical_prompt="Casual low-effort acknowledgment or let the turn conclude naturally.",
                max_words=6,
                max_bubbles=1,
                forbidden_approaches=["writing a new essay", "forcing small talk"],
                effort_parity_directive="Under 5 words.",
            )

        if act == "agreement":
            return ResponseStrategy(
                strategy_name="short_acknowledgment",
                tactical_prompt="Brief agreement or reinforcement ('wahi na', 'exactly', 'fr').",
                max_words=8,
                max_bubbles=1,
                forbidden_approaches=["repeating their whole point", "formal validation"],
                effort_parity_directive="Under 6 words.",
            )

        if act == "disagreement":
            return ResponseStrategy(
                strategy_name="playful_counter",
                tactical_prompt="State your casual counter-opinion playfully or deadpan.",
                max_words=12,
                max_bubbles=2,
                forbidden_approaches=["angry debate", "formal essays"],
                effort_parity_directive="Under 12 words.",
            )

        if act == "venting":
            return ResponseStrategy(
                strategy_name="supportive_response",
                tactical_prompt=(
                    "The user is stressed or venting. Validate them like a close friend ('arre yaar', 'fml sahi me', 'itna kyu padhna hai'). "
                    "Do NOT act like a corporate therapist or give unsolicited advice unless asked."
                ),
                max_words=20,
                max_bubbles=2,
                forbidden_approaches=["corporate empathy", "'I understand your feelings'", "toxic positivity", "unsolicited advice lists"],
                effort_parity_directive="Relatable, friendly response under 20 words.",
            )

        if act == "compliment":
            return ResponseStrategy(
                strategy_name="match_excitement",
                tactical_prompt="Accept compliment with self-deprecating wit or casual thanks ('hehe thankss', 'obvious hai na').",
                max_words=10,
                max_bubbles=1,
                forbidden_approaches=["excessive modesty", "stiff formal thank you"],
                effort_parity_directive="Under 8 words.",
            )

        if act in ("question_personal", "question_logistical"):
            return ResponseStrategy(
                strategy_name="direct_answer",
                tactical_prompt="Answer the question directly and casually in 1-2 bubbles. Stay relevant.",
                max_words=14,
                max_bubbles=2,
                forbidden_approaches=["rambling about homework unprompted", "answering with rhetorical self-questions"],
                effort_parity_directive="Direct answer under 14 words.",
            )

        if act == "serious_insult":
            return ResponseStrategy(
                strategy_name="calm_response",
                tactical_prompt="Stay deadpan, unbothered, or dismissive ('kya problem hai bhai', 'chill kar'). Do not escalate aggressively.",
                max_words=8,
                max_bubbles=1,
                forbidden_approaches=["crying", "apologizing profusely", "getting into an abusive brawl"],
                effort_parity_directive="Deadpan response under 8 words.",
            )

        # Fallback default
        max_w = min(max(incoming_word_count * 2, 8), 16)
        return ResponseStrategy(
            strategy_name="direct_answer",
            tactical_prompt="Respond conversationally and naturally under 12 words.",
            max_words=max_w,
            max_bubbles=2,
            forbidden_approaches=["essay replies", "formal English in Hinglish chat"],
            effort_parity_directive=f"Keep reply length under {max_w} words.",
        )

    @classmethod
    def _build_strategy_object(
        cls,
        strategy_name: str,
        intent: SocialIntent,
        incoming_word_count: int,
    ) -> ResponseStrategy:
        """Construct ResponseStrategy from a known strategy name."""
        if strategy_name == "playful_counter":
            return ResponseStrategy(
                strategy_name="playful_counter",
                tactical_prompt="Counter playfully with a short witty tease under 8 words. Do NOT explain why.",
                max_words=8,
                max_bubbles=2,
                forbidden_approaches=["literal explanation", "apologizing", "echoing question"],
                effort_parity_directive="Under 8 words.",
            )
        if strategy_name == "short_acknowledgment":
            return ResponseStrategy(
                strategy_name="short_acknowledgment",
                tactical_prompt="Send a short acknowledgment under 5 words.",
                max_words=5,
                max_bubbles=1,
                forbidden_approaches=["long replies"],
                effort_parity_directive="Under 5 words.",
            )
        if strategy_name == "supportive_response":
            return ResponseStrategy(
                strategy_name="supportive_response",
                tactical_prompt="Relatable, friendly response without corporate therapy tropes.",
                max_words=18,
                max_bubbles=2,
                forbidden_approaches=["corporate empathy", "unsolicited advice lists"],
                effort_parity_directive="Under 18 words.",
            )
        return ResponseStrategy(
            strategy_name=strategy_name,
            tactical_prompt=f"Respond using strategy '{strategy_name}' under 12 words.",
            max_words=12,
            max_bubbles=2,
            forbidden_approaches=[],
            effort_parity_directive="Under 12 words.",
        )
