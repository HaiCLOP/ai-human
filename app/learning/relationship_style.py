"""Dyadic relationship dynamics analyzer modeling interaction patterns between operator and contacts."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.learning.models import RelationshipStyleProfile

logger = get_logger("learning.relationship_style")


class RelationshipStyleAnalyzer:
    """Analyzes observable interaction dynamics across conversations between operator and a contact."""

    def analyze(
        self,
        contact_id: str,
        all_conversation_messages: list[dict[str, Any]],
        operator_name: str,
    ) -> RelationshipStyleProfile:
        """Analyze dyadic relationship dynamics from all exchanged messages."""
        total = len(all_conversation_messages)
        if total == 0:
            return RelationshipStyleProfile(
                contact_id=contact_id,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

        op_lower = operator_name.strip().lower()
        operator_msgs = [m for m in all_conversation_messages if m.get("sender_name", "").strip().lower() == op_lower]
        contact_msgs = [m for m in all_conversation_messages if m.get("sender_name", "").strip().lower() != op_lower]

        op_count = len(operator_msgs)
        contact_count = len(contact_msgs)
        op_message_ratio = round(op_count / total, 3)

        # 1. Conversation Initiation & Clustering
        # Cluster messages by 3-hour inactivity windows into distinct conversation sessions
        conversations: list[list[dict[str, Any]]] = []
        current_cluster: list[dict[str, Any]] = []

        sorted_msgs = sorted(all_conversation_messages, key=lambda m: m.get("timestamp_ms", 0))
        for m in sorted_msgs:
            if not current_cluster:
                current_cluster.append(m)
            else:
                prev_ts = current_cluster[-1].get("timestamp_ms", 0)
                curr_ts = m.get("timestamp_ms", 0)
                if curr_ts - prev_ts > 10800000:  # 3 hours gap
                    conversations.append(current_cluster)
                    current_cluster = [m]
                else:
                    current_cluster.append(m)

        if current_cluster:
            conversations.append(current_cluster)

        # Who initiated each conversation?
        operator_initiations = 0
        exchange_lengths: list[int] = []
        for conv in conversations:
            if conv:
                first_sender = conv[0].get("sender_name", "").strip().lower()
                if first_sender == op_lower:
                    operator_initiations += 1
                exchange_lengths.append(len(conv))

        total_convs = max(1, len(conversations))
        operator_initiation_ratio = round(operator_initiations / total_convs, 3)
        avg_exchange_length = round(statistics.mean(exchange_lengths), 1) if exchange_lengths else 4.0

        # 2. Reel sharing frequency
        reel_count = sum(1 for m in all_conversation_messages if m.get("message_type") == "REEL")
        reel_sharing_freq = round(reel_count / total, 3)

        # 3. Behavioral dimensions: Playfulness, Sarcasm, Teasing, Seriousness
        # Measured through conversational markers
        all_texts = [m.get("text", "").lower() for m in all_conversation_messages if m.get("text")]
        combined_text = " ".join(all_texts)

        laugh_hits = sum(combined_text.count(w) for w in ["😂", "😭", "haha", "lmao", "lol", "dead"])
        sarcasm_hits = sum(combined_text.count(w) for w in ["sahi hai", "accha", "achha", "kya baat", "great", "wah"])
        teasing_hits = sum(combined_text.count(w) for w in ["paka mat", "chup", "gadhe", "pagal", "kuch bhi", "sharam kar"])
        serious_hits = sum(combined_text.count(w) for w in ["exam", "test", "serious", "marks", "deadline", "urgent", "problem"])

        playfulness = round(min(0.95, max(0.10, (laugh_hits / max(1, len(all_texts))) * 2.0)), 2)
        sarcasm = round(min(0.95, max(0.05, (sarcasm_hits / max(1, len(all_texts))) * 2.5)), 2)
        teasing = round(min(0.90, max(0.05, (teasing_hits / max(1, len(all_texts))) * 3.0)), 2)
        seriousness = round(min(0.90, max(0.05, (serious_hits / max(1, len(all_texts))) * 2.0)), 2)

        # 4. Response style classification
        if avg_exchange_length >= 15 and playfulness >= 0.5:
            response_style = "rapid_banter"
        elif total_convs <= 3 and total < 20:
            response_style = "infrequent_catchup"
        elif reel_sharing_freq >= 0.25:
            response_style = "meme_heavy_sharing"
        else:
            response_style = "asynchronous_casual"

        confidence = round(min(0.99, total / (total + 20)), 3)

        return RelationshipStyleProfile(
            contact_id=contact_id,
            playfulness=playfulness,
            sarcasm=sarcasm,
            teasing=teasing,
            seriousness=seriousness,
            operator_initiation_ratio=operator_initiation_ratio,
            operator_message_ratio=op_message_ratio,
            avg_exchange_length=avg_exchange_length,
            reel_sharing_frequency=reel_sharing_freq,
            response_style=response_style,
            confidence=confidence,
            evidence_count=total,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
