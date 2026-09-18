"""Tests for Social Intent and Speech-Act Analysis Engine."""

import pytest
from app.conversation.social_intent import SocialIntentAnalyzer


def test_banter_insult_why_are_you_so_dumb():
    text = "Why are you so dumb"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "playful_insult"
    assert intent.playfulness >= 0.7
    assert intent.hostility < 0.2
    assert intent.seriousness < 0.2
    assert intent.expected_reply_length == "very_short"
    assert intent.requires_response is True


def test_banter_insult_hinglish_laugh():
    text = "tu pagal hai kya 😭"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "playful_insult"
    assert intent.playfulness >= 0.8
    assert intent.hostility < 0.2
    assert intent.expected_reply_length == "very_short"


def test_serious_hostile_attack():
    text = "I hate you, get lost and shut up bitch"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "serious_insult"
    assert intent.hostility >= 0.8
    assert intent.seriousness >= 0.7


def test_pure_laugh_reaction():
    for laugh in ["haha", "lmao", "ded", "dead", "😭", "😂😂"]:
        intent = SocialIntentAnalyzer.analyze(laugh)
        assert intent.social_act == "reaction_laugh"
        assert intent.playfulness >= 0.8
        assert intent.expected_reply_length == "very_short"


def test_teasing():
    text = "bade log ameer ho gaye flex mat kar"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "teasing"
    assert intent.playfulness >= 0.8
    assert intent.expected_reply_length == "very_short"


def test_acknowledgment():
    for ack in ["ok", "acha", "hmm", "haan", "cool", "mast", "badhiya"]:
        intent = SocialIntentAnalyzer.analyze(ack)
        assert intent.social_act == "acknowledgment"
        assert intent.expected_reply_length == "very_short"


def test_greeting():
    for greet in ["yo", "hey", "hello", "kaha hai tu", "kya scene"]:
        intent = SocialIntentAnalyzer.analyze(greet)
        assert intent.social_act == "greeting"
        assert intent.expected_reply_length == "very_short"


def test_venting():
    text = "fml test me hag diya bohot dimag kharab ho gaya"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "venting"
    assert intent.emotional_intensity >= 0.7
    assert intent.expected_reply_length == "medium"


def test_farewell():
    text = "so raha hu bye gn"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "farewell"
    assert intent.expected_reply_length == "very_short"


def test_compliment():
    text = "looking pretty cute in this story slay"
    intent = SocialIntentAnalyzer.analyze(text)
    assert intent.social_act == "compliment"
    assert intent.playfulness >= 0.5
