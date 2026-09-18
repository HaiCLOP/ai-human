"""Social Intent and Speech-Act Analysis Engine.

Deterministically analyzes conversational intent, hostility, playfulness, seriousness,
and expected reply length to eliminate socially awkward or overly explanatory responses.
"""

from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("conversation.social_intent")


class SocialIntent(BaseModel):
    """Rich semantic and pragmatic representation of an incoming conversational turn."""

    social_act: str = Field(
        default="other",
        description="Core speech act category (e.g. playful_insult, greeting, venting, acknowledgment).",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    seriousness: float = Field(default=0.2, ge=0.0, le=1.0, description="0.0 = total joke, 1.0 = grave/critical")
    hostility: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 = friendly/banter, 1.0 = hostile/attacking")
    playfulness: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0 = deadpan/dry/serious, 1.0 = high banter/playful")
    emotional_intensity: float = Field(default=0.3, ge=0.0, le=1.0)
    urgency: float = Field(default=0.1, ge=0.0, le=1.0)
    expected_reply_length: Literal["very_short", "short", "medium", "long"] = Field(
        default="short",
        description="Target reply brevity. Banter/insults/greetings must be very_short or short.",
    )
    requires_response: bool = Field(default=True)
    detected_signals: list[str] = Field(default_factory=list)


class SocialIntentAnalyzer:
    """Classifies incoming text into speech-acts and pragmatic dimensions."""

    # Emojis indicating laughter / playfulness
    LAUGH_EMOJIS = {"😭", "💀", "😂", "🤣", "😆"}
    LAUGH_TEXT_PATTERNS = [
        r"\b(ha(ha)+|lmao+|lol+|rofl+|ded|dead|xd+|lmfao+)\b",
    ]

    # Hostility signals
    HOSTILE_PATTERNS = [
        r"\b(i hate you|get lost|fuck off|f\*\*k|die|shut up bitch|piss off|i will hurt)\b",
    ]

    # Tease / banter insult patterns
    PLAYFUL_INSULT_PATTERNS = [
        r"\bwhy (are )?you so (dumb|slow|stupid|idiot|annoying|weird|cringe|lame)\b",
        r"\b(you('re|r| are) (so )?(dumb|slow|stupid|an idiot|a loser|a clown|noob))\b",
        r"\b(tu (itna|itni)? (dumb|slow|pagal|gadha|gadhi|bewakoof) kyu hai)\b",
        r"\b(tu (pagal|gadha|chutiya|slow|bewakoof) hai kya)\b",
        r"\b(dumb|slow|pagal|gadha|dumbo|noob|clown) (hai|ho|kahin ka|kahin ki|saala|saali)\b",
        r"\b(dimag (kharab mat kar|ghutne me|nahi hai tera|hai kya tera))\b",
        r"\b(brain ?dead|slow speed|zero iq)\b",
        r"\b(chup kar|bakwaas band kar|shakal dekh apni)\b",
    ]

    TEASING_PATTERNS = [
        r"\b(bade log|hero ban raha|ameer log|rich kid|flex mat kar|tu to rehne hi de)\b",
        r"\b(itna attitude|bhav mat kha|kareebi|overacting|dramaqueen)\b",
    ]

    COMPLIMENT_PATTERNS = [
        r"\b(mast lag|badhiya lag|looking (good|great|pretty|cute|fire)|smart|slay|killing it|proud of you|op|banger|w rizz)\b",
        r"\b(congrats|congratulations|mubarak|shabaash|well done)\b",
    ]

    GREETING_PATTERNS = [
        r"^(yo|hey+|hello+|hi+|hii+|hiii+|sup|wassup|ola)\b",
        r"\b(kya haal|kya scene|aur bata|aur batao|kaha hai|kaha ho|sab badhiya)\b",
    ]

    FAREWELL_PATTERNS = [
        r"\b(bye+|byee+|good ?night|gn|tata|cya|chalta hu|chalti hu|so raha hu|so rahi hu|kal milte|see you)\b",
    ]

    ACKNOWLEDGMENT_PATTERNS = [
        r"^(ok+|okay+|acha+|achha+|accha+|hmm+|haan+|ha+|hnn+|cool|k|theek hai|thik hai|sahi|got it|done|alright|fine|mast|badhiya)\b",
    ]

    AGREEMENT_PATTERNS = [
        r"\b(fr|facts|true|bilkul|sahi bola|sahi baat|wahi to|exactly|100%|same here|real|ditto)\b",
    ]

    DISAGREEMENT_PATTERNS = [
        r"\b(nah|nahi yaar|galat|no way|aisa nahi hai|bilkul nahi|nope|false|not true)\b",
    ]

    VENTING_PATTERNS = [
        r"\b(fml|mood kharab|itna padhna|bore ho gaya|bore ho gayi|dimag kharab ho gaya|hate this|so done|tired|thak gaya|thak gayi|mar gaya|mar gayi|bohot stress|stress ho raha)\b",
        r"\b(paper kharab gaya|test me hag diya|fail ho jaunga|fail ho jaungi|daant padi)\b",
    ]

    QUESTION_LOGISTICAL_PATTERNS = [
        r"\b(kab|kaha milna|time kya hua|kitne baje|notes bhej|link de|pdf bhej|kahan aana hai)\b",
        r"\b(what time|when|where should we|send the (notes|link|pdf|doc))\b",
    ]

    QUESTION_PERSONAL_PATTERNS = [
        r"\b(tu bata|kya kar raha|kya kar rahi|kaisa hai|kaisi hai|kya chal raha|kya socha)\b",
        r"\b(what (do you think|are you doing|happened)|how are you)\b",
    ]

    SHOCK_PATTERNS = [
        r"\b(what\?+|wtf|aree+|hain\?+|wait what|seriously\?+|no wayyy+|bhaiii+)\b",
    ]

    APOLOGY_PATTERNS = [
        r"\b(sorry+|sry|maaf kar|maaf karna|my bad|galti ho gayi)\b",
    ]

    GRATITUDE_PATTERNS = [
        r"\b(thanks+|thank you|ty|shukriya|dhanyawad|thx)\b",
    ]

    URGENCY_PATTERNS = [
        r"\b(urgent|call utha|jaldi kar|fast|emergency|asap|pick up|jaldi reply kar)\b",
    ]

    CHECK_IN_PATTERNS = [
        r"\b(kaha gayab hai|sab theek|are you alive|alive\?|kaha mar gaye|kaha mar gayi)\b",
    ]

    CLARIFICATION_PATTERNS = [
        r"\b(kya matlab|samjha nahi|samjhi nahi|what do you mean|huh\?+|meaning\?)\b",
    ]

    @classmethod
    def analyze(
        cls,
        text: str,
        context_history: list[dict[str, str]] | None = None,
        contact_id: str | None = None,
    ) -> SocialIntent:
        """Analyze message text and optional conversation history into SocialIntent."""
        raw_text = text.strip()
        lower = raw_text.lower()
        signals: list[str] = []

        # Check laugh cues
        has_laugh_emoji = any(e in raw_text for e in cls.LAUGH_EMOJIS)
        has_laugh_text = any(re.search(p, lower) for p in cls.LAUGH_TEXT_PATTERNS)
        is_laughing = has_laugh_emoji or has_laugh_text

        # Check hostility
        is_hostile = any(re.search(p, lower) for p in cls.HOSTILE_PATTERNS)
        if is_hostile and not is_laughing:
            signals.append("hostility_markers")
            return SocialIntent(
                social_act="serious_insult",
                confidence=0.9,
                seriousness=0.85,
                hostility=0.85,
                playfulness=0.0,
                emotional_intensity=0.8,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        words = lower.split()

        # Standalone short acknowledgment check (ok, hmm, acha, mast, badhiya)
        if len(words) <= 2 and any(re.search(p, lower) for p in cls.ACKNOWLEDGMENT_PATTERNS):
            signals.append("acknowledgment_pattern")
            return SocialIntent(
                social_act="acknowledgment",
                confidence=0.9,
                seriousness=0.2,
                hostility=0.0,
                playfulness=0.3,
                expected_reply_length="very_short",
                requires_response=False,
                detected_signals=signals,
            )

        # Venting check (takes precedence over banter/slang like dimag)
        if any(re.search(p, lower) for p in cls.VENTING_PATTERNS):
            signals.append("venting_pattern")
            return SocialIntent(
                social_act="venting",
                confidence=0.85,
                seriousness=0.75,
                hostility=0.1,
                playfulness=0.15,
                emotional_intensity=0.8,
                expected_reply_length="medium",
                requires_response=True,
                detected_signals=signals,
            )

        # 1. Check for Playful Insult / Teasing (CRITICAL: "Why are you so dumb", "tu pagal hai kya")
        # Notice: rhetorical questions like "why are you so dumb" MUST match playful_insult BEFORE question checks!
        is_playful_insult = any(re.search(p, lower) for p in cls.PLAYFUL_INSULT_PATTERNS)
        is_teasing = any(re.search(p, lower) for p in cls.TEASING_PATTERNS)

        if is_playful_insult:
            signals.append("playful_insult_pattern")
            if is_hostile and not is_laughing:
                return SocialIntent(
                    social_act="serious_insult",
                    confidence=0.85,
                    seriousness=0.8,
                    hostility=0.85,
                    playfulness=0.1,
                    emotional_intensity=0.7,
                    expected_reply_length="short",
                    requires_response=True,
                    detected_signals=signals + ["hostility_markers"],
                )
            # Standard banter / tease insult:
            playfulness = 0.9 if is_laughing else 0.8
            return SocialIntent(
                social_act="playful_insult",
                confidence=0.9,
                seriousness=0.1,
                hostility=0.1,
                playfulness=playfulness,
                emotional_intensity=0.5,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals + (["laugh_markers"] if is_laughing else []),
            )

        if is_teasing:
            signals.append("teasing_pattern")
            return SocialIntent(
                social_act="teasing",
                confidence=0.85,
                seriousness=0.15,
                hostility=0.05,
                playfulness=0.85,
                emotional_intensity=0.4,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 2. Pure laughter reaction
        words = lower.split()
        if is_laughing and (len(words) <= 3 or len(re.sub(r"[a-z0-9]", "", lower)) >= len(lower) // 2):
            signals.append("reaction_laugh")
            return SocialIntent(
                social_act="reaction_laugh",
                confidence=0.95,
                seriousness=0.05,
                hostility=0.0,
                playfulness=0.9,
                emotional_intensity=0.6,
                expected_reply_length="very_short",
                requires_response=False,  # Can laugh back or stay short
                detected_signals=signals,
            )

        # 3. Urgency
        if any(re.search(p, lower) for p in cls.URGENCY_PATTERNS):
            signals.append("urgency_pattern")
            return SocialIntent(
                social_act="urgency",
                confidence=0.9,
                seriousness=0.85,
                hostility=0.1,
                playfulness=0.0,
                emotional_intensity=0.8,
                urgency=0.9,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 4. Venting
        if any(re.search(p, lower) for p in cls.VENTING_PATTERNS):
            signals.append("venting_pattern")
            return SocialIntent(
                social_act="venting",
                confidence=0.85,
                seriousness=0.75,
                hostility=0.1,
                playfulness=0.15,
                emotional_intensity=0.8,
                expected_reply_length="medium",
                requires_response=True,
                detected_signals=signals,
            )

        # 5. Farewell
        if any(re.search(p, lower) for p in cls.FAREWELL_PATTERNS):
            signals.append("farewell_pattern")
            return SocialIntent(
                social_act="farewell",
                confidence=0.9,
                seriousness=0.3,
                hostility=0.0,
                playfulness=0.3,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 6. Greeting
        if any(re.search(p, lower) for p in cls.GREETING_PATTERNS):
            signals.append("greeting_pattern")
            return SocialIntent(
                social_act="greeting",
                confidence=0.9,
                seriousness=0.2,
                hostility=0.0,
                playfulness=0.5,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 7. Check-in
        if any(re.search(p, lower) for p in cls.CHECK_IN_PATTERNS):
            signals.append("check_in_pattern")
            return SocialIntent(
                social_act="check_in",
                confidence=0.85,
                seriousness=0.4,
                hostility=0.0,
                playfulness=0.5,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 8. Compliment
        if any(re.search(p, lower) for p in cls.COMPLIMENT_PATTERNS):
            signals.append("compliment_pattern")
            return SocialIntent(
                social_act="compliment",
                confidence=0.85,
                seriousness=0.4,
                hostility=0.0,
                playfulness=0.6,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        # 9. Gratitude
        if any(re.search(p, lower) for p in cls.GRATITUDE_PATTERNS):
            signals.append("gratitude_pattern")
            return SocialIntent(
                social_act="gratitude",
                confidence=0.9,
                seriousness=0.3,
                hostility=0.0,
                playfulness=0.3,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 10. Apology
        if any(re.search(p, lower) for p in cls.APOLOGY_PATTERNS):
            signals.append("apology_pattern")
            return SocialIntent(
                social_act="apology",
                confidence=0.85,
                seriousness=0.6,
                hostility=0.0,
                playfulness=0.2,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 11. Shock
        if any(re.search(p, lower) for p in cls.SHOCK_PATTERNS):
            signals.append("shock_pattern")
            return SocialIntent(
                social_act="reaction_shock",
                confidence=0.85,
                seriousness=0.4,
                hostility=0.0,
                playfulness=0.6,
                emotional_intensity=0.7,
                expected_reply_length="very_short",
                requires_response=True,
                detected_signals=signals,
            )

        # 12. Clarification
        if any(re.search(p, lower) for p in cls.CLARIFICATION_PATTERNS):
            signals.append("clarification_pattern")
            return SocialIntent(
                social_act="clarification",
                confidence=0.85,
                seriousness=0.5,
                hostility=0.0,
                playfulness=0.3,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        # 13. Agreement
        if any(re.search(p, lower) for p in cls.AGREEMENT_PATTERNS):
            signals.append("agreement_pattern")
            return SocialIntent(
                social_act="agreement",
                confidence=0.85,
                seriousness=0.3,
                hostility=0.0,
                playfulness=0.4,
                expected_reply_length="very_short",
                requires_response=False,
                detected_signals=signals,
            )

        # 14. Disagreement
        if any(re.search(p, lower) for p in cls.DISAGREEMENT_PATTERNS):
            signals.append("disagreement_pattern")
            return SocialIntent(
                social_act="disagreement",
                confidence=0.8,
                seriousness=0.5,
                hostility=0.1,
                playfulness=0.4,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        # 15. Acknowledgment (ok, hmm, acha, haan, cool)
        if any(re.search(p, lower) for p in cls.ACKNOWLEDGMENT_PATTERNS) and len(words) <= 3:
            signals.append("acknowledgment_pattern")
            return SocialIntent(
                social_act="acknowledgment",
                confidence=0.9,
                seriousness=0.2,
                hostility=0.0,
                playfulness=0.3,
                expected_reply_length="very_short",
                requires_response=False,
                detected_signals=signals,
            )

        # 16. Questions (Logistical vs Personal)
        if any(re.search(p, lower) for p in cls.QUESTION_LOGISTICAL_PATTERNS):
            signals.append("question_logistical_pattern")
            return SocialIntent(
                social_act="question_logistical",
                confidence=0.85,
                seriousness=0.6,
                hostility=0.0,
                playfulness=0.2,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        if any(re.search(p, lower) for p in cls.QUESTION_PERSONAL_PATTERNS) or "?" in raw_text:
            signals.append("question_personal_pattern")
            return SocialIntent(
                social_act="question_personal",
                confidence=0.75,
                seriousness=0.4,
                hostility=0.0,
                playfulness=0.4,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        # 17. Shared Content / Reel link
        if "instagram.com" in lower or "reel" in lower or "http" in lower:
            signals.append("sharing_content_pattern")
            return SocialIntent(
                social_act="sharing_content",
                confidence=0.9,
                seriousness=0.2,
                hostility=0.0,
                playfulness=0.6,
                expected_reply_length="short",
                requires_response=True,
                detected_signals=signals,
            )

        # 18. Default fallback: evaluate length and punctuation
        length_category: Literal["very_short", "short", "medium", "long"] = "short"
        if len(words) <= 3:
            length_category = "very_short"
        elif len(words) > 15:
            length_category = "medium"

        return SocialIntent(
            social_act="other",
            confidence=0.5,
            seriousness=0.3,
            hostility=0.0,
            playfulness=0.4,
            emotional_intensity=0.3,
            expected_reply_length=length_category,
            requires_response=True,
            detected_signals=["default_fallback"],
        )
