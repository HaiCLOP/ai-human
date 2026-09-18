"""Instagram export JSON normalizer into clean internal representation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.learning.models import (
    MessageType,
    NormalizedConversation,
    NormalizedMessage,
    ReactionItem,
    HistoricalTurn,
)

logger = get_logger("learning.normalizer")


def fix_instagram_encoding(val: str) -> str:
    """Fix Instagram JSON export character encoding quirk (Latin-1 misinterpreted UTF-8)."""
    if not val or not isinstance(val, str):
        return ""
    try:
        return val.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return val


class InstagramNormalizer:
    """Normalizes raw Instagram JSON conversation exports into canonical data structures."""

    @staticmethod
    def classify_message_type(msg: dict[str, Any]) -> MessageType:
        """Classify message type according to Instagram export attributes."""
        # 1. Calls
        if "call_duration" in msg:
            return MessageType.CALL

        # 2. Shared content / Reels / Stories
        share_obj = msg.get("share")
        if isinstance(share_obj, dict):
            link = share_obj.get("link", "") or ""
            if "reel" in link.lower() or "reels" in link.lower():
                return MessageType.REEL
            if "stories" in link.lower() or "story" in link.lower():
                return MessageType.STORY
            return MessageType.REEL if "instagram.com" in link else MessageType.ATTACHMENT

        # 3. Audio files
        if msg.get("audio_files"):
            return MessageType.AUDIO

        # 4. Photos
        if msg.get("photos"):
            return MessageType.PHOTO

        # 5. Videos
        if msg.get("videos"):
            return MessageType.VIDEO

        # 6. Standalone reactions
        if not msg.get("content") and msg.get("reactions"):
            return MessageType.REACTION

        # 7. Text content
        content = msg.get("content")
        if content is not None:
            # Check for story reaction or mention in content
            content_lower = str(content).lower()
            if "reacted to your story" in content_lower or "shared a story" in content_lower:
                return MessageType.STORY
            if "sent an attachment" in content_lower:
                return MessageType.ATTACHMENT
            return MessageType.TEXT

        return MessageType.UNKNOWN

    @classmethod
    def normalize_message(
        cls,
        raw_msg: dict[str, Any],
        conversation_id: str,
        sender_id_map: dict[str, str] | None = None,
    ) -> NormalizedMessage:
        """Convert a single raw message dict into NormalizedMessage."""
        raw_sender = fix_instagram_encoding(raw_msg.get("sender_name", "Unknown"))
        ts_ms = int(raw_msg.get("timestamp_ms", 0))
        if ts_ms > 0:
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            ts_iso = dt.isoformat()
        else:
            ts_iso = datetime.now(timezone.utc).isoformat()

        msg_type = cls.classify_message_type(raw_msg)
        text = fix_instagram_encoding(raw_msg.get("content", "") or "")

        # Extract reactions
        reactions: list[ReactionItem] = []
        raw_reactions = raw_msg.get("reactions", [])
        if isinstance(raw_reactions, list):
            for r in raw_reactions:
                if isinstance(r, dict):
                    reactions.append(
                        ReactionItem(
                            reaction=fix_instagram_encoding(r.get("reaction", "")),
                            actor=fix_instagram_encoding(r.get("actor", "")),
                        )
                    )

        # Extract shared media metadata
        shared_url: str | None = None
        share_text: str | None = None
        owner: str | None = None

        share_obj = raw_msg.get("share")
        if isinstance(share_obj, dict):
            shared_url = share_obj.get("link")
            share_text = fix_instagram_encoding(share_obj.get("share_text", "") or "") or None
            owner = fix_instagram_encoding(share_obj.get("original_content_owner", "") or "") or None

        if not text and share_text:
            text = share_text

        sender_id = sender_id_map.get(raw_sender, raw_sender) if sender_id_map else raw_sender

        return NormalizedMessage(
            conversation_id=conversation_id,
            sender_id=sender_id,
            sender_display_name=raw_sender,
            timestamp_ms=ts_ms,
            timestamp_iso=ts_iso,
            message_type=msg_type,
            text=text,
            shared_url=shared_url,
            share_text=share_text,
            original_content_owner=owner,
            reactions=reactions,
            metadata={
                "is_unsent": raw_msg.get("is_unsent", False),
                "is_geoblocked": raw_msg.get("is_geoblocked_for_viewer", False),
            },
        )

    @classmethod
    def normalize_conversation(
        cls,
        file_path: Path | str,
        conversation_id: str | None = None,
        operator_names: list[str] | None = None,
    ) -> NormalizedConversation:
        """Parse and normalize an entire Instagram conversation JSON export file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        cid = conversation_id or path.stem

        # Extract participants
        raw_participants = data.get("participants", [])
        participants: list[str] = []
        if isinstance(raw_participants, list):
            for p in raw_participants:
                if isinstance(p, dict) and "name" in p:
                    participants.append(fix_instagram_encoding(p["name"]))
                elif isinstance(p, str):
                    participants.append(fix_instagram_encoding(p))

        # Identify operator and contact if possible
        operator_name: str | None = None
        contact_name: str | None = None
        if operator_names:
            op_set = {op.lower().strip() for op in operator_names}
            for p in participants:
                if p.lower().strip() in op_set:
                    operator_name = p
                else:
                    contact_name = p

        # Parse messages
        raw_messages = data.get("messages", [])
        normalized_messages: list[NormalizedMessage] = []
        if isinstance(raw_messages, list):
            for raw_m in raw_messages:
                if isinstance(raw_m, dict):
                    normalized_messages.append(cls.normalize_message(raw_m, conversation_id=cid))

        # Instagram stores messages in reverse-chronological order (newest first)
        # Sort chronologically (oldest first)
        normalized_messages.sort(key=lambda m: m.timestamp_ms)

        # Set sequential message_index and sender_role
        op_set = {op.lower().strip() for op in operator_names} if operator_names else set()
        if operator_name:
            op_set.add(operator_name.lower().strip())

        for idx, m in enumerate(normalized_messages):
            m.message_index = idx
            s_clean = m.sender_display_name.lower().strip()
            if s_clean in op_set:
                m.sender_role = "operator"
            elif contact_name and s_clean == contact_name.lower().strip():
                m.sender_role = "contact"
            elif op_set:
                m.sender_role = "contact"
            else:
                m.sender_role = "unknown"

        date_start = normalized_messages[0].timestamp_iso if normalized_messages else None
        date_end = normalized_messages[-1].timestamp_iso if normalized_messages else None

        return NormalizedConversation(
            conversation_id=cid,
            file_path=str(path.resolve()),
            participants=participants,
            operator_name=operator_name,
            contact_name=contact_name,
            messages=normalized_messages,
            message_count=len(normalized_messages),
            date_start=date_start,
            date_end=date_end,
        )

    @classmethod
    def extract_contextual_turns(
        cls,
        normalized_conv: NormalizedConversation,
        operator_name: str | None = None,
        contact_id: str | None = None,
    ) -> list[HistoricalTurn]:
        """Extract paired contextual turns (contact prompt -> operator response) with speech-act annotations."""
        from app.conversation.social_intent import SocialIntentAnalyzer

        turns: list[HistoricalTurn] = []
        messages = normalized_conv.messages
        if not messages:
            return turns

        cid = contact_id or normalized_conv.contact_name or "contact"
        op_lower = (operator_name or normalized_conv.operator_name or "").strip().lower()

        i = 0
        n = len(messages)
        context_history: list[dict[str, str]] = []

        while i < n:
            m = messages[i]
            is_op = (m.sender_role == "operator") or (bool(op_lower) and m.sender_display_name.strip().lower() == op_lower)

            if not is_op:
                contact_block: list[NormalizedMessage] = []
                while i < n:
                    curr = messages[i]
                    curr_is_op = (curr.sender_role == "operator") or (bool(op_lower) and curr.sender_display_name.strip().lower() == op_lower)
                    if curr_is_op:
                        break
                    contact_block.append(curr)
                    i += 1

                operator_block: list[NormalizedMessage] = []
                while i < n:
                    curr = messages[i]
                    curr_is_op = (curr.sender_role == "operator") or (bool(op_lower) and curr.sender_display_name.strip().lower() == op_lower)
                    if not curr_is_op:
                        break
                    operator_block.append(curr)
                    i += 1

                c_text = " \n ".join(m.text.strip() for m in contact_block if m.text and m.text.strip())
                o_text = " \n ".join(m.text.strip() for m in operator_block if m.text and m.text.strip())

                if c_text and o_text:
                    intent = SocialIntentAnalyzer.analyze(c_text)
                    words = o_text.split()
                    w_count = len(words)

                    if w_count <= 3:
                        len_cat = "very_short"
                    elif w_count <= 10:
                        len_cat = "short"
                    elif w_count <= 25:
                        len_cat = "medium"
                    else:
                        len_cat = "long"

                    # Infer strategy
                    if intent.social_act == "playful_insult":
                        strategy = "playful_counter"
                    elif intent.social_act == "teasing":
                        strategy = "playful_counter"
                    elif intent.social_act == "reaction_laugh":
                        strategy = "short_acknowledgment"
                    elif intent.social_act in ("greeting", "farewell", "acknowledgment"):
                        strategy = "short_acknowledgment"
                    elif intent.social_act == "venting":
                        strategy = "supportive_response"
                    elif intent.social_act == "compliment":
                        strategy = "match_excitement"
                    elif len_cat == "very_short":
                        strategy = "short_acknowledgment"
                    else:
                        strategy = "direct_answer"

                    energy = 0.5
                    if "!" in o_text or any(e in o_text for e in ["😭", "😂", "🤣"]):
                        energy = 0.8
                    elif w_count <= 2:
                        energy = 0.3

                    turn = HistoricalTurn(
                        turn_id=f"turn_{normalized_conv.conversation_id}_{contact_block[0].timestamp_ms}_{len(turns)}",
                        conversation_id=normalized_conv.conversation_id,
                        contact_id=cid,
                        contact_text=c_text,
                        operator_text=o_text,
                        timestamp_ms=contact_block[-1].timestamp_ms,
                        social_act=intent.social_act,
                        social_intent=intent.model_dump(),
                        response_strategy=strategy,
                        response_length_category=len_cat,
                        context_messages=list(context_history[-3:]),
                        turn_energy=energy,
                    )
                    turns.append(turn)

                for msg in contact_block:
                    if msg.text and msg.text.strip():
                        context_history.append({"sender": "contact", "text": msg.text.strip()})
                for msg in operator_block:
                    if msg.text and msg.text.strip():
                        context_history.append({"sender": "operator", "text": msg.text.strip()})
                if len(context_history) > 10:
                    context_history = context_history[-10:]
            else:
                if m.text and m.text.strip():
                    context_history.append({"sender": "operator", "text": m.text.strip()})
                i += 1

        return turns
