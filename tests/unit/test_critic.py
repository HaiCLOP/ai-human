"""Tests for Response Quality Critic."""

import pytest
from app.conversation.critic import ResponseQualityCritic
from app.conversation.social_intent import SocialIntentAnalyzer


def test_critic_flags_the_original_failure_case():
    incoming = "Why are you so dumb"
    intent = SocialIntentAnalyzer.analyze(incoming)
    failing_reply = "dumb? nahi slow speed hai bas tu bhi same level pe ha"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=failing_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="playful_counter",
    )
    assert not res.passes
    assert "rhetorical_insult_echo" in res.detected_issues
    assert "defensive_justification_on_banter" in res.detected_issues
    assert res.suggested_repair is not None


def test_critic_passes_authentic_counter_banter():
    incoming = "Why are you so dumb"
    intent = SocialIntentAnalyzer.analyze(incoming)
    good_reply = "tu konsa topper hai waise"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=good_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="playful_counter",
    )
    assert res.passes
    assert res.score >= 0.8


def test_critic_flags_forbidden_skull_emoji():
    incoming = "haha funny"
    intent = SocialIntentAnalyzer.analyze(incoming)
    reply = "dead lmao 💀"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=reply,
        incoming_text=incoming,
        intent=intent,
    )
    assert not res.passes
    assert "forbidden_skull_emoji" in res.detected_issues


def test_critic_flags_ai_tells_and_corporate_empathy():
    incoming = "fml test kharab gaya"
    intent = SocialIntentAnalyzer.analyze(incoming)
    reply = "I understand your distress. To ensure optimal results, next time you should prepare a strict timetable."

    res = ResponseQualityCritic.evaluate(
        candidate_reply=reply,
        incoming_text=incoming,
        intent=intent,
        strategy="supportive_response",
    )
    assert not res.passes
    assert any("ai_tell_pattern" in iss for iss in res.detected_issues)


def test_critic_flags_effort_parity_violation():
    incoming = "Mast"
    intent = SocialIntentAnalyzer.analyze(incoming)
    long_reply = "Haan bilkul mast hona bhi chahiye! Aur batao tumhara din kaisa raha? Mera to bohot busy tha."

    res = ResponseQualityCritic.evaluate(
        candidate_reply=long_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="short_acknowledgment",
    )
    assert not res.passes
    assert any("effort_parity" in iss for iss in res.detected_issues)


def test_critic_flags_unsolicited_self_disclosure_on_short_acknowledgment():
    incoming = "Okayyy"
    intent = SocialIntentAnalyzer.analyze(incoming)
    failing_reply = "bas thoda chill music sun rahi hu thoda snack khati thi"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=failing_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="react_casual",
    )
    assert not res.passes
    assert "unsolicited_self_disclosure_on_acknowledgment" in res.detected_issues
    assert res.suggested_repair == "haan"


def test_critic_flags_temporal_inconsistency_on_ongoing_events():
    incoming = "Naa 7 baje hogi"
    intent = SocialIntentAnalyzer.analyze(incoming)
    failing_reply = "theek hai shayad 7 baje hogi \n film kaisi thi mast thi?"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=failing_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="continue_topic",
    )
    assert not res.passes
    assert "temporal_inconsistency_past_tense_on_ongoing_event" in res.detected_issues
    assert res.suggested_repair == "achha theek hai tu dekh le fir"


def test_critic_flags_unsolicited_topic_on_time_query():
    incoming = "What's the time"
    intent = SocialIntentAnalyzer.analyze(incoming)
    failing_reply = "abhi 6:33 baje hain, film khatam ho gaya?"

    res = ResponseQualityCritic.evaluate(
        candidate_reply=failing_reply,
        incoming_text=incoming,
        intent=intent,
        strategy="direct_answer",
    )
    assert not res.passes
    assert "unsolicited_topic_on_logistical_query" in res.detected_issues
