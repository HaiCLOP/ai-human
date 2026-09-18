"""Master Conversation Orchestrator tying cognitive layers to routine, availability, and LLM generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.ai.persona import CharacterProfile, load_character_profile
from app.ai.prompts import CharacterResponsePlan, PromptBuilder
from app.ai.provider import LLMProvider
from app.ai.router import get_llm_provider
from app.ai.validator import ResponseValidator, ValidationResult
from app.conversation.action_planner import ActionPlanner
from app.conversation.chemistry import ChemistryModel
from app.conversation.critic import ResponseQualityCritic
from app.conversation.response_strategy import ResponseStrategySelector
from app.conversation.situation_retriever import SituationAwareRetriever
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.state import (
    ConversationState,
    compute_message_fingerprint,
    update_conversation_state,
)
from app.core.logging import bind_correlation_id, get_logger
from app.humor.engine import HumorEngine
from app.memory.manager import MemoryManager
from app.memory.style import StyleLearner
from app.rag.retriever import LocalRAGRetriever
from app.routine.life_events import LifeEventManager
from app.routine.manager import RoutineManager
from app.routine.models import AvailabilityState
from app.routine.mood import MoodEngine
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import (
    AuditRepository,
    ConversationRepository,
    MessageRepository,
)

logger = get_logger("conversation.manager")


class ConversationManager:
    """End-to-end conversation orchestrator with deterministic availability gating and life simulation."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        llm_provider: LLMProvider | None = None,
        character_profile: CharacterProfile | None = None,
        rag_retriever: LocalRAGRetriever | None = None,
        routine_manager: RoutineManager | None = None,
    ):
        self.db = db or get_db_manager()
        self.conv_repo = ConversationRepository(self.db)
        self.msg_repo = MessageRepository(self.db)
        self.audit_repo = AuditRepository(self.db)

        self.memory_mgr = MemoryManager(self.db)
        self.style_learner = StyleLearner(self.db)
        self.humor_engine = HumorEngine(self.db)
        self.chemistry_model = ChemistryModel(self.db)
        self.rag_retriever = rag_retriever or LocalRAGRetriever(self.db)

        self.llm_provider = llm_provider or get_llm_provider()
        self.profile = character_profile or load_character_profile()
        self.validator = ResponseValidator()

        from app.memory.repetition import TopicSaturationCache
        self.repetition_cache = TopicSaturationCache()

        from app.storage.historical_repo import HistoricalRepository
        self.hist_repo = HistoricalRepository(self.db)
        self.situation_retriever = SituationAwareRetriever(self.db, repo=self.hist_repo)
        self.conversation_states: dict[str, ConversationState] = {}

        # Life simulation & Routine subsystems
        self.routine_mgr = routine_manager or RoutineManager(
            db=self.db,
            routine_config=self.profile.routine,
            departure_warning_minutes=self.profile.availability.departure_warning_minutes,
        )
        self.life_mgr = LifeEventManager(self.db)
        if self.profile.academics:
            self.life_mgr.sync_academics_from_config(self.profile.academics)

    async def handle_incoming_message(
        self,
        conversation_id: str,
        sender_handle: str,
        message_text: str,
        correlation_id: str | None = None,
        simulated_time: datetime | None = None,
        force_available: bool = False,
    ) -> str | None:
        """Process inbound message through complete cognitive cycle, obeying real-time availability."""
        cid = correlation_id or str(uuid.uuid4())
        bind_correlation_id(cid)
        now = simulated_time or datetime.now(timezone.utc)

        clean_text = message_text.strip()
        if not clean_text:
            return None

        # 1. Idempotency Check
        fingerprint = compute_message_fingerprint(
            conversation_id=conversation_id,
            sender_handle=sender_handle,
            message_text=clean_text,
            timestamp=now,
        )

        if self.msg_repo.is_already_handled(fingerprint):
            logger.info(
                "message.duplicate_ignored",
                conversation_id=conversation_id,
                fingerprint=fingerprint[:12],
            )
            return None

        # 2. Register Inbound Message in Database
        self.conv_repo.get_or_create(
            conversation_id=conversation_id,
            participant_handle=sender_handle,
        )

        inbound_record = self.msg_repo.record_incoming_message(
            conversation_id=conversation_id,
            fingerprint=fingerprint,
            sender_handle=sender_handle,
            content=clean_text,
        )

        # 3. PRE-LLM AVAILABILITY GATEKEEPER
        avail_state, current_activity = self.routine_mgr.resolve_availability(now)
        departure_decision = self.routine_mgr.evaluate_departure(now)

        # If character is already away / busy and NOT in the act of departing right now:
        if not force_available and avail_state != AvailabilityState.AVAILABLE and not (departure_decision and departure_decision.should_depart_now):
            logger.info(
                "conversation.character_unavailable_message_queued",
                state=avail_state.value,
                activity=current_activity.activity,
                away_until=current_activity.end,
            )
            # Message remains in 'RECEIVED' state. Do NOT generate response or call LLM.
            return None

        self.msg_repo.update_status(inbound_record.message_id, "PROCESSING")

        logger.info(
            "message.detected",
            conversation_id=conversation_id,
            sender=sender_handle,
            length=len(clean_text),
        )

        try:
            # 4. Asynchronous Cognitive Updates
            self.style_learner.observe_message(conversation_id, clean_text)
            self.memory_mgr.extract_and_persist_facts(conversation_id, clean_text)
            chemistry_notes = self.chemistry_model.evaluate_and_update(conversation_id, clean_text)
            humor_decision = self.humor_engine.evaluate_humor(conversation_id, clean_text)

            # 5. Check for Accumulated Messages While Away (Batch Return Check)
            history = self.msg_repo.get_recent_messages(conversation_id, limit=20)
            accumulated_unhandled = [
                m for m in history
                if m.sender_type == "USER" and m.status == "RECEIVED" and m.message_id != inbound_record.message_id
            ]

            batched_texts: list[str] = []
            if accumulated_unhandled:
                for m in accumulated_unhandled:
                    batched_texts.append(f"{m.sender_handle}: {m.content}")
                # Append current message as latest
                batched_texts.append(f"{sender_handle}: {clean_text}")
                logger.info("conversation.batching_accumulated_messages", count=len(batched_texts))

            # 6. Check Upcoming Commitment Warning & Departure Directives
            upcoming = self.routine_mgr.check_upcoming_commitment(now)
            upcoming_warning = None
            if upcoming:
                upcoming_warning = (
                    f"Your commitment '{upcoming.activity}' begins in {upcoming.minutes_until_start} minutes "
                    f"at {upcoming.location}. You may casually mention this in conversation if appropriate."
                )

            departure_directive = None
            if departure_decision and departure_decision.should_depart_now:
                departure_directive = (
                    f"You must leave the conversation NOW. Reason: {departure_decision.departure_reason}. "
                    f"Say a natural goodbye, explain where you are going (e.g. tuition, class, sleep, dinner), "
                    f"and mention you will return in approximately {departure_decision.estimated_away_hours} hours. "
                    f"Be in character, slightly dramatic or exhausted if appropriate."
                )

            # 7. Compute Dynamic Mood with 11-dimension model and per-conversation persistence
            academic_context = self.life_mgr.get_academic_context()
            effective_avail = AvailabilityState.AVAILABLE if force_available else avail_state
            mood = MoodEngine.compute_mood(
                conversation_id=conversation_id,
                current_activity=current_activity.activity,
                availability=effective_avail,
                upcoming_exams=self.life_mgr.academic_repo.get_upcoming_exams(),
                pending_homework=self.life_mgr.academic_repo.get_pending_homework(),
            )
            emotional_state_notes = MoodEngine.to_prompt_descriptor(mood)

            # 8. Context Assembly
            style_notes = self.style_learner.get_style_notes_for_prompt(conversation_id)
            relevant_memories = self.memory_mgr.get_relevant_memories(conversation_id, clean_text)
            rag_chunks = self.rag_retriever.retrieve(clean_text)

            routine_context = (
                f"Current Time: {now.strftime('%A %H:%M')}, Activity: {current_activity.activity}, "
                f"Location: {current_activity.location}, State: {effective_avail.value}"
            )

            # Context Budgeting: Only include academic notes if contextually provoked, departing, or batch return
            user_lower = clean_text.lower()
            should_include_academics = any(
                w in user_lower for w in ["padh", "study", "exam", "test", "hw", "homework", "coaching", "tuition", "marks", "syllabus"]
            ) or bool(departure_directive) or bool(batched_texts)

            filtered_academic_notes = (
                academic_context.upcoming_exams_summary + academic_context.pending_homework_summary
                if should_include_academics
                else ()
            )

            # Topic Cooldown & Repetition Prevention
            saturated_topics = self.repetition_cache.get_saturated_topics()
            saturated_openers = self.repetition_cache.recent_openers[-2:]

            # 8b. Historical Intelligence Context Assembly
            op_style_notes: list[str] = []
            contact_style_notes: list[str] = []
            rel_notes: list[str] = []
            hist_mem_notes: list[str] = []

            try:
                op_profile = self.hist_repo.get_operator_style()
                if op_profile:
                    op_style_notes.append(f"Operator lowercase preference: {int(op_profile.get('lowercase_ratio', 0.9) * 100)}%")
                    op_style_notes.append(f"Operator Hinglish ratio: {int(op_profile.get('hinglish_ratio', 0.5) * 100)}%")
                    if op_profile.get("slang_frequencies_json"):
                        import json as json_lib
                        slangs = list(json_lib.loads(op_profile["slang_frequencies_json"]).keys())[:6]
                        if slangs:
                            op_style_notes.append(f"Frequent slang words in this social circle: {', '.join(slangs)}")

                from app.learning.operator_detector import OperatorDetector
                contact_id = OperatorDetector.generate_contact_id(sender_handle)
                c_style = self.hist_repo.get_contact_style(contact_id)
                if c_style:
                    contact_style_notes.append(f"Contact brevity: {c_style.get('brevity_level', 'medium')}")
                    if c_style.get("sarcasm_score", 0) >= 0.4:
                        contact_style_notes.append("Contact enjoys sarcasm and dry deadpan responses")

                rel_style = self.hist_repo.get_relationship_style(contact_id)
                if rel_style:
                    rel_notes.append(f"Playfulness level: {rel_style.get('playfulness', 0.5):.2f}, Style: {rel_style.get('response_style', 'casual')}")

                top_mems = self.hist_repo.get_top_memories(contact_id, limit=3)
                for tm in top_mems:
                    hist_mem_notes.append(f"[{tm.get('category')}] {tm.get('statement')}")
            except Exception as e:
                logger.debug("conversation.historical_context_assembly_failed", error=str(e))

            # 8c. Social Intent & Response Strategy Analysis
            from app.learning.operator_detector import OperatorDetector
            contact_id = OperatorDetector.generate_contact_id(sender_handle)
            social_intent = SocialIntentAnalyzer.analyze(clean_text)
            # Attach raw text for state tracking
            social_intent._raw_text = clean_text  # type: ignore[attr-defined]

            current_conv_state = self.conversation_states.get(
                conversation_id,
                ConversationState(conversation_id=conversation_id, contact_id=contact_id),
            )

            rel_style_dict = self.hist_repo.get_relationship_style(contact_id)
            behavioral_pattern = self.hist_repo.get_pattern_for_act(social_intent.social_act, contact_id)

            selected_strategy = ResponseStrategySelector.select_strategy(
                intent=social_intent,
                relationship_profile=rel_style_dict,
                conversation_mode=current_conv_state.social_mode,
                behavioral_pattern=behavioral_pattern,
                incoming_word_count=len(clean_text.split()),
            )

            # 8d. Action Planning — WHAT Vesper should do (runs before prompt assembly)
            planned_action_result = ActionPlanner.plan(
                state=current_conv_state,
                intent=social_intent,
                relationship_profile=rel_style_dict,
                mood=mood,
            )
            planned_action_str = (
                f"Action: {planned_action_result.action.value}\n"
                f"Hint: {planned_action_result.tactical_hint}\n"
                f"Brevity: {planned_action_result.brevity_target}\n"
                f"Multi-bubble allowed: {'yes' if planned_action_result.allow_multi_bubble else 'no'}"
            )

            logger.info(
                "action.planned",
                action=planned_action_result.action.value,
                confidence=planned_action_result.confidence,
                social_act=social_intent.social_act,
            )

            # Retrieve situationally relevant historical examples
            historical_turn_examples = self.situation_retriever.retrieve_examples(
                incoming_text=clean_text,
                intent=social_intent,
                contact_id=contact_id,
                limit=3,
            )

            # Build tactical prompt notes
            social_intent_notes = [
                f"Detected Speech Act: {social_intent.social_act} (confidence: {social_intent.confidence:.2f})",
                f"Playfulness: {social_intent.playfulness:.2f}, Hostility: {social_intent.hostility:.2f}, Seriousness: {social_intent.seriousness:.2f}",
                f"Target reply brevity: {social_intent.expected_reply_length}",
            ]
            response_strategy_notes = [
                f"Strategy: {selected_strategy.strategy_name}",
                f"Tactical Directive: {selected_strategy.tactical_prompt}",
                f"Word Budget: Max {selected_strategy.max_words} words across at most {selected_strategy.max_bubbles} bubbles.",
            ]
            if selected_strategy.forbidden_approaches:
                response_strategy_notes.append(
                    f"FORBIDDEN: {', '.join(selected_strategy.forbidden_approaches)}"
                )
            if selected_strategy.effort_parity_directive:
                response_strategy_notes.append(f"Effort Parity: {selected_strategy.effort_parity_directive}")

            behavioral_pattern_notes = []
            if behavioral_pattern:
                behavioral_pattern_notes.append(
                    f"Historical tendency for {social_intent.social_act}: typical reply is {behavioral_pattern.get('typical_length', 'short')} (confidence {behavioral_pattern.get('confidence', 0.5):.2f})"
                )

            # Build conversation state notes for prompt
            conv_state_notes: list[str] = []
            if current_conv_state.current_topic:
                conv_state_notes.append(f"Active topic: {current_conv_state.current_topic}")
            if current_conv_state.social_mode != "casual":
                conv_state_notes.append(f"Current mode: {current_conv_state.social_mode}")
            if current_conv_state.active_joke:
                conv_state_notes.append(f"Active joke thread: {current_conv_state.active_joke}")
            if current_conv_state.unanswered_question:
                conv_state_notes.append(f"Pending unanswered question: {current_conv_state.unanswered_question}")
            if current_conv_state.consecutive_short_replies >= 2:
                conv_state_notes.append(f"You've given {current_conv_state.consecutive_short_replies} consecutive very short replies — vary your response if natural.")

            system_instruction = PromptBuilder.build_system_instruction(self.profile)
            user_prompt = PromptBuilder.build_prompt(
                current_message=clean_text if not batched_texts else "",
                user_handle=sender_handle,
                conversation_history=history,
                user_style_notes=style_notes,
                chemistry_notes=chemistry_notes,
                relevant_memories=relevant_memories,
                rag_context=rag_chunks,
                humor_directive=humor_decision.directive_text,
                routine_context=routine_context,
                upcoming_warning=upcoming_warning,
                departure_directive=departure_directive,
                emotional_state_notes=emotional_state_notes,
                academic_notes=filtered_academic_notes,
                batched_messages_while_away=batched_texts,
                saturated_topics=saturated_topics,
                saturated_openers=saturated_openers,
                operator_style_notes=op_style_notes,
                contact_style_notes=contact_style_notes,
                relationship_notes=rel_notes,
                historical_memories=hist_mem_notes,
                social_intent_notes=social_intent_notes,
                response_strategy_notes=response_strategy_notes,
                historical_examples=historical_turn_examples,
                behavioral_pattern_notes=behavioral_pattern_notes,
                planned_action=planned_action_str,
                conversation_state_notes=conv_state_notes,
            )

            # 9. LLM Generation
            logger.info("llm.request_started", provider=type(self.llm_provider).__name__)
            llm_response = await self.llm_provider.generate(
                prompt=user_prompt,
                system_instruction=system_instruction,
                response_schema=CharacterResponsePlan,
            )

            structured = llm_response.structured_data or {}
            intent = structured.get("intent", "REPLY")
            candidate_reply = structured.get("reply_text", "")
            raw_bubbles = structured.get("bubbles", [])

            # If bubbles are provided by the LLM, join them or use them
            if raw_bubbles and isinstance(raw_bubbles, list) and len(raw_bubbles) > 0:
                candidate_reply = " \n ".join(b.strip() for b in raw_bubbles if b.strip())
            elif not candidate_reply and raw_bubbles:
                candidate_reply = " \n ".join(str(b) for b in raw_bubbles)

            # 9b. Response Quality Critic
            critic_res = ResponseQualityCritic.evaluate(
                candidate_reply=candidate_reply,
                incoming_text=clean_text,
                intent=social_intent,
                strategy=selected_strategy.strategy_name,
            )

            if not critic_res.passes:
                logger.warning(
                    "response.critic_flagged_issues",
                    issues=critic_res.detected_issues,
                    score=critic_res.score,
                    candidate=candidate_reply,
                )

                # SINGLE REGENERATION ATTEMPT before falling back to suggested_repair
                repair_prompt = (
                    f"[REPAIR DIRECTIVE]\n"
                    f"Your previous reply was flagged. Issues: {', '.join(critic_res.detected_issues)}.\n"
                    f"Regenerate a new reply that avoids all these issues while following the planned action.\n"
                    f"Original message: {clean_text}\n\n"
                ) + user_prompt

                try:
                    logger.info("response.critic_retry_started")
                    retry_response = await self.llm_provider.generate(
                        prompt=repair_prompt,
                        system_instruction=system_instruction,
                        response_schema=CharacterResponsePlan,
                    )
                    retry_structured = retry_response.structured_data or {}
                    retry_reply = retry_structured.get("reply_text", "")
                    retry_bubbles = retry_structured.get("bubbles", [])
                    if retry_bubbles and isinstance(retry_bubbles, list):
                        retry_reply = " \n ".join(b.strip() for b in retry_bubbles if b.strip())

                    retry_critic = ResponseQualityCritic.evaluate(
                        candidate_reply=retry_reply,
                        incoming_text=clean_text,
                        intent=social_intent,
                        strategy=selected_strategy.strategy_name,
                    )

                    if retry_reply and (retry_critic.passes or retry_critic.score > critic_res.score):
                        candidate_reply = retry_reply
                        logger.info("response.critic_retry_accepted", score=retry_critic.score)
                    elif critic_res.suggested_repair:
                        logger.info("response.critic_applied_repair", repair=critic_res.suggested_repair)
                        candidate_reply = critic_res.suggested_repair
                except Exception as retry_err:
                    logger.warning("response.critic_retry_failed", error=str(retry_err))
                    if critic_res.suggested_repair:
                        candidate_reply = critic_res.suggested_repair

            if intent == "SILENCE" or not candidate_reply:
                logger.info("conversation.character_chose_silence")
                self.msg_repo.update_status(inbound_record.message_id, "IGNORED")
                return None

            # 10. Validation & Safety
            recent_character_replies = [m.content for m in history if m.sender_type == "CHARACTER"]
            validation_result = self.validator.validate(
                candidate_text=candidate_reply,
                recent_replies=recent_character_replies,
            )

            if not validation_result.is_valid:
                logger.warning("response.validation_failed", reason=validation_result.rejection_reason)
                self.msg_repo.update_status(inbound_record.message_id, "FAILED")
                self.audit_repo.record_event(
                    event_type="VALIDATION_REJECTION",
                    severity="WARNING",
                    component="validator",
                    payload={"reason": validation_result.rejection_reason, "reply_text": candidate_reply},
                    correlation_id=cid,
                )
                return None

            approved_reply = validation_result.sanitized_text

            # Register in repetition cache
            self.repetition_cache.register_character_message(approved_reply)

            # 10b. Apply mood event from the interaction
            try:
                if social_intent.playfulness >= 0.6:
                    MoodEngine.apply_event(conversation_id, "banter", magnitude=0.10)
                elif social_intent.hostility >= 0.5:
                    MoodEngine.apply_event(conversation_id, "insult_hostile", magnitude=0.12)
                elif social_intent.social_act == "venting":
                    MoodEngine.apply_event(conversation_id, "sad_topic", magnitude=0.08)
            except Exception:
                pass  # mood event failure is non-critical

            # 11. Update Conversation State Post-Send
            self.conversation_states[conversation_id] = update_conversation_state(
                current_conv_state,
                social_intent,
                selected_strategy.strategy_name,
                vesper_reply=approved_reply,
            )

            # 12. State Updates Post-Send
            self.msg_repo.record_character_message(
                conversation_id=conversation_id,
                content=approved_reply,
                character_handle=self.profile.identity.handle,
            )
            self.msg_repo.update_status(inbound_record.message_id, "SENT")

            # If messages were batched, mark all accumulated messages as SENT
            for m in accumulated_unhandled:
                self.msg_repo.update_status(m.message_id, "SENT")

            # If this was a departure, enforce away state in database
            if departure_decision and departure_decision.should_depart_now:
                self.routine_mgr.mark_departure(
                    character_id="default",
                    activity=departure_decision.activity,
                    new_state=avail_state,
                    location=current_activity.location,
                    away_until_iso=departure_decision.away_until_iso,
                )

            logger.info("response.approved", conversation_id=conversation_id, intent=intent)
            return approved_reply

        except Exception as e:
            logger.error("conversation.processing_error", error=str(e))
            self.msg_repo.update_status(inbound_record.message_id, "FAILED")
            raise

    async def initiate_conversation(
        self,
        conversation_id: str,
        target_handle: str,
        custom_message: str | None = None,
        outreach_directive: str | None = None,
        force_available: bool = False,
        simulated_time: datetime | None = None,
    ) -> str | None:
        """Proactively initiate a conversation turn with a target user."""
        now = simulated_time or datetime.now(timezone.utc)

        # 1. Availability check unless force_available
        avail_state, current_activity = self.routine_mgr.resolve_availability(now)
        if not force_available and avail_state != AvailabilityState.AVAILABLE:
            logger.warning(
                "conversation.outreach_blocked_by_availability",
                state=avail_state.value,
                activity=current_activity.activity,
                target=target_handle,
            )
            return None

        # 2. Register Conversation
        self.conv_repo.get_or_create(
            conversation_id=conversation_id,
            participant_handle=target_handle,
        )

        if custom_message:
            validation_result = self.validator.validate(custom_message)
            if not validation_result.is_valid:
                logger.warning("conversation.custom_outreach_rejected", reason=validation_result.rejection_reason)
                return None
            approved = validation_result.sanitized_text
            self.msg_repo.record_character_message(
                conversation_id=conversation_id,
                content=approved,
                character_handle=self.profile.identity.handle,
            )
            return approved

        # 3. Dynamic cognitive and context assembly
        history = self.msg_repo.get_recent_messages(conversation_id, limit=12)
        style_notes = self.style_learner.get_style_notes_for_prompt(conversation_id)
        chemistry_notes = self.chemistry_model.evaluate_and_update(conversation_id, "")
        humor_decision = self.humor_engine.evaluate_humor(conversation_id, "")
        relevant_memories = self.memory_mgr.get_relevant_memories(conversation_id, "start conversation")
        rag_chunks = self.rag_retriever.retrieve(f"{target_handle} friend conversation")

        academic_context = self.life_mgr.get_academic_context()
        effective_avail = AvailabilityState.AVAILABLE if force_available else avail_state
        mood = MoodEngine.compute_mood(
            conversation_id=conversation_id,
            current_activity=current_activity.activity,
            availability=effective_avail,
            upcoming_exams=self.life_mgr.academic_repo.get_upcoming_exams(),
            pending_homework=self.life_mgr.academic_repo.get_pending_homework(),
        )
        emotional_state_notes = MoodEngine.to_prompt_descriptor(mood)

        routine_context = (
            f"Current Time: {now.strftime('%A %H:%M')}, Activity: {current_activity.activity}, "
            f"Location: {current_activity.location}, State: {effective_avail.value}"
        )

        default_outreach = (
            outreach_directive
            or f"Initiate a casual, chill DM to {target_handle}. Ask what they are up to, or mention something relatable (movies, music, weekend plans, memes). Do NOT talk about homework, exams, or syllabus unless asked."
        )

        system_instruction = PromptBuilder.build_system_instruction(self.profile)
        user_prompt = PromptBuilder.build_prompt(
            current_message="",
            user_handle=target_handle,
            conversation_history=history,
            user_style_notes=style_notes,
            chemistry_notes=chemistry_notes,
            relevant_memories=relevant_memories,
            rag_context=rag_chunks,
            humor_directive=humor_decision.directive_text,
            routine_context=routine_context,
            emotional_state_notes=emotional_state_notes,
            academic_notes=(),
            outreach_directive=default_outreach,
        )

        # 4. LLM Generation
        logger.info("llm.request_started", provider=type(self.llm_provider).__name__, mode="outreach")
        llm_response = await self.llm_provider.generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            response_schema=CharacterResponsePlan,
        )

        structured = llm_response.structured_data or {}
        candidate_reply = structured.get("reply_text", "")
        raw_bubbles = structured.get("bubbles", [])
        if raw_bubbles and isinstance(raw_bubbles, list) and len(raw_bubbles) > 0:
            candidate_reply = " \n ".join(b.strip() for b in raw_bubbles if b.strip())

        if not candidate_reply:
            return None

        # 5. Validation
        recent_character_replies = [m.content for m in history if m.sender_type == "CHARACTER"]
        validation_result = self.validator.validate(
            candidate_text=candidate_reply,
            recent_replies=recent_character_replies,
        )
        if not validation_result.is_valid:
            logger.warning("conversation.outreach_validation_failed", reason=validation_result.rejection_reason)
            return None

        approved_reply = validation_result.sanitized_text

        # Register in repetition cache
        self.repetition_cache.register_character_message(approved_reply)

        # 6. Save in DB
        self.msg_repo.record_character_message(
            conversation_id=conversation_id,
            content=approved_reply,
            character_handle=self.profile.identity.handle,
        )
        logger.info("conversation.outreach_sent", target=target_handle, reply=approved_reply)
        return approved_reply
