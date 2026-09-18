"""Unit tests for Vesper Conversational Intelligence V3.

Verifies the 12 core test cases specified in Section 37:
1. "Mast hai" -> No unnecessary question; natural reaction.
2. "Naa 7 baje hogi" -> Movie remains ongoing; no past tense question.
3. "wyd" with physics study state -> Reflects physics, no generic filler.
4. "kyu?" -> Actual grounded explanation, not "padhai bas".
5. "okayy" -> Very short acknowledgment.
6. "bro tu pagal hai" -> Counter-tease / playful deflection, not defensive.
7. "what happened today" -> Specific life details, not generic AI deflection.
8. "haan" -> No unsolicited self-disclosure.
9. "acha" -> No unrelated monologue or interrogations.
10. Long emotional message -> Higher effort supportive validation.
11. Ongoing event -> No premature past-tense probe.
12. Similar social situation -> Behavioral retrieval match.
"""

from datetime import datetime, timezone
import pytest

from app.conversation.critic import ResponseQualityCritic
from app.conversation.policy import ConversationPolicy, ContributionType
from app.conversation.situation_retriever import SituationAwareRetriever
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.state import ConversationState, update_conversation_state
from app.routine.models import VesperLifeState
from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository


@pytest.fixture
def physics_life_state():
    return VesperLifeState(
        activity="Studying Physics",
        location="study desk",
        current_task="solving thermodynamics numericals",
        energy=0.45,
        mood_label="exhausted",
        free_time=False,
        next_event="dinner at 21:00",
        unfinished_thought="why is thermodynamics formula so confusing",
    )


def test_1_mast_hai_no_unnecessary_question():
    """User gives a short positive reaction 'Mast hai'. Vesper must react/agree without asking questions."""
    incoming = "Mast hai"
    intent = SocialIntentAnalyzer.analyze(incoming)
    assert intent.social_act == "reaction_short"

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert plan.contribution_type == ContributionType.REACTION_ONLY
    assert plan.question_budget == 0
    assert not plan.should_ask_question

    # Candidate with question fails
    bad_reply = "Mast laga na? Kaisa tha climax?"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "question_budget_exceeded" in critic_bad.detected_issues

    # Authentic reaction passes
    good_reply = "sahi hai na"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_2_naa_7_baje_hogi_ongoing_event_tracking():
    """User says 'Naa 7 baje hogi'. System tracks movie as ongoing until 19:00, refuses past-tense query."""
    incoming = "Naa 7 baje hogi"
    intent = SocialIntentAnalyzer.analyze(incoming)
    assert intent.social_act == "schedule_future_completion"

    state = ConversationState(conversation_id="conv_movie", contact_id="user1")
    state.current_topic = "movie"
    state = update_conversation_state(
        state=state,
        social_intent=intent,
        strategy_name="direct_answer",
        vesper_reply="theek hai",
        current_user_text=incoming,
    )

    # Verify state tracking
    check_time = datetime(2026, 9, 18, 18, 30, tzinfo=timezone.utc)
    assert state.is_subject_ongoing("movie", check_time)

    # Policy must keep question budget 0
    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent, state=state)
    assert plan.question_budget == 0

    # Past-tense probe fails
    bad_reply = "achha film kaisi thi?"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        state=state,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "temporal_inconsistency_past_tense_on_ongoing_event" in critic_bad.detected_issues


def test_3_wyd_authoritative_life_state(physics_life_state):
    """User asks 'wyd'. Response must reflect physics numericals from life state, not generic filler."""
    incoming = "wyd"
    intent = SocialIntentAnalyzer.analyze(incoming)

    plan = ConversationPolicy.evaluate(
        clean_text=incoming,
        social_intent=intent,
        life_state=physics_life_state,
    )
    assert plan.contribution_type == ContributionType.SELF_DISCLOSURE
    assert "physics" in plan.information_to_add.lower() or "numerical" in plan.information_to_add.lower()

    # Vague generic filler fails Critic
    bad_reply = "kuch nahi bas chill kar rahi hu"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "contextual_vagueness" in critic_bad.detected_issues

    # Specific grounded reply passes
    good_reply = "thermodynamics ke numericals dekh rahi hu dimag kharab ho raha"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_4_kyu_probe_explanation(physics_life_state):
    """User asks 'kyu?'. Vesper must give a real grounded explanation, not 'padhai bas'."""
    incoming = "kyu?"
    intent = SocialIntentAnalyzer.analyze(incoming)

    plan = ConversationPolicy.evaluate(
        clean_text=incoming,
        social_intent=intent,
        life_state=physics_life_state,
    )
    assert plan.contribution_type == ContributionType.STORY_DETAIL

    # Vague non-answer fails
    bad_reply = "bas padhai"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "contextual_vagueness" in critic_bad.detected_issues

    # Grounded explanation passes
    good_reply = "kal test hai aur numericals ek bhi solve nahi ho rahe 😭"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_5_okayy_very_short_acknowledgment():
    """User says 'okayy'. Vesper must send a very short closer with zero questions."""
    incoming = "okayy"
    intent = SocialIntentAnalyzer.analyze(incoming)

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert plan.contribution_type == ContributionType.CLOSE
    assert plan.question_budget == 0
    assert plan.desired_length == "very_short"

    # Multi-sentence interrogation fails
    bad_reply = "haan theek hai! aur batao kya kar rahe ho abhi?"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes

    # Single-word acknowledgment passes
    good_reply = "haan"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_6_bro_tu_pagal_hai_counter_banter():
    """User banters 'bro tu pagal hai'. Vesper must counter-tease or deflect playfully, not be defensive."""
    incoming = "bro tu pagal hai"
    intent = SocialIntentAnalyzer.analyze(incoming)
    assert intent.social_act in ("playful_insult", "teasing", "clarification")

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert plan.contribution_type in (ContributionType.COUNTER_TEASE, ContributionType.PLAYFUL_DEFLECTION)

    # Defensive justification fails
    bad_reply = "Actually mai pagal nahi hu bas thoda tired hu."
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "defensive_justification_on_banter" in critic_bad.detected_issues or "unnatural_ending_period" in critic_bad.detected_issues

    # Playful counter passes
    good_reply = "shakal dekh apni pehle 😭"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_7_what_happened_today_specific_life_detail(physics_life_state):
    """User asks 'what happened today'. Vesper shares specific life details, not generic AI deflection."""
    incoming = "what happened today"
    intent = SocialIntentAnalyzer.analyze(incoming)

    plan = ConversationPolicy.evaluate(
        clean_text=incoming,
        social_intent=intent,
        life_state=physics_life_state,
    )
    assert plan.allow_self_disclosure

    # Robotic assistant trope fails
    bad_reply = "As an AI, nothing really happened today. How can I help you?"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert any("ai_tell_pattern" in iss for iss in critic_bad.detected_issues)

    # Specific school/study detail passes
    good_reply = "school me practical me daant pad gayi aur abhi physics ka homework bacha hai"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_8_haan_no_unsolicited_self_disclosure():
    """User says 'haan'. Vesper must not volunteer unprompted updates about her activities."""
    incoming = "haan"
    intent = SocialIntentAnalyzer.analyze(incoming)
    assert intent.social_act == "acknowledgment"

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert not plan.allow_self_disclosure

    # Unprompted self-update fails Critic
    bad_reply = "bas thoda music sun rahi hu aur assignments kar rahi hu"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "unsolicited_self_disclosure_on_acknowledgment" in critic_bad.detected_issues

    # Simple closure passes
    good_reply = "sahi hai"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_9_acha_no_unrelated_monologue():
    """User says 'acha'. Vesper gives brief acknowledgment, never monologue or ask 2 questions."""
    incoming = "acha"
    intent = SocialIntentAnalyzer.analyze(incoming)

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert plan.question_budget == 0

    # Overanswering with monologue fails
    bad_reply = "Haan bilkul mast hona bhi chahiye! Aur batao tumhara din kaisa raha? Mera to bohot busy tha."
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes
    assert "question_budget_exceeded" in critic_bad.detected_issues

    # Brief reaction passes
    good_reply = "haan"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_10_long_emotional_venting():
    """User vents about a terrible test. Vesper responds with higher effort and supportive validation."""
    incoming = "fml test me hag diya bohot bura lag raha hai"
    intent = SocialIntentAnalyzer.analyze(incoming)
    assert intent.social_act == "venting"

    plan = ConversationPolicy.evaluate(clean_text=incoming, social_intent=intent)
    assert plan.desired_effort in ("medium", "high")
    assert plan.contribution_type == ContributionType.SUPPORT

    # Corporate advice fails Critic
    bad_reply = "I understand your distress. Next time prepare a strict timetable."
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert not critic_bad.passes

    # Warm relatable validation passes
    good_reply = "arre yaar fml itna bura tha kya test"
    critic_good = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        contribution_plan=plan,
    )
    assert critic_good.passes


def test_11_ongoing_event_no_past_tense_probe():
    """When an event is ongoing, Critic rejects any past-tense question about it."""
    incoming = "abhi movie dekh raha hu"
    intent = SocialIntentAnalyzer.analyze(incoming)

    state = ConversationState(conversation_id="movie_test")
    state.current_topic = "movie"

    # Past-tense probe fails
    bad_reply = "kaisi thi film?"
    critic_bad = ResponseQualityCritic.evaluate(
        candidate_reply=bad_reply,
        incoming_text=incoming,
        intent=intent,
        state=state,
    )
    assert not critic_bad.passes
    assert "temporal_inconsistency_past_tense_on_ongoing_event" in critic_bad.detected_issues


def test_12_situation_retrieval_matching():
    """SituationAwareRetriever retrieves relevant operator examples for social acts."""
    db = get_db_manager()
    repo = HistoricalRepository(db)
    retriever = SituationAwareRetriever(db, repo=repo)

    incoming = "Why are you so dumb"
    intent = SocialIntentAnalyzer.analyze(incoming)
    examples = retriever.retrieve_examples(incoming_text=incoming, intent=intent, limit=2)

    assert len(examples) > 0
    top_ex = examples[0]
    assert "contact_text" in top_ex
    assert "operator_text" in top_ex
    assert top_ex["score"] > 0.4
