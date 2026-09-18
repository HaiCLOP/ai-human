"""Tests for Response Strategy Selector."""

import pytest
from app.conversation.response_strategy import ResponseStrategySelector
from app.conversation.social_intent import SocialIntentAnalyzer


def test_strategy_playful_insult():
    intent = SocialIntentAnalyzer.analyze("Why are you so dumb")
    strat = ResponseStrategySelector.select_strategy(intent)
    assert strat.strategy_name == "playful_counter"
    assert strat.max_words <= 8
    assert "literal explanation" in strat.forbidden_approaches
    assert "dumb?" in strat.tactical_prompt or "teasing" in strat.tactical_prompt


def test_strategy_teasing():
    intent = SocialIntentAnalyzer.analyze("bade log")
    strat = ResponseStrategySelector.select_strategy(intent)
    assert strat.strategy_name == "playful_counter"
    assert strat.max_words <= 10


def test_strategy_acknowledgment():
    intent = SocialIntentAnalyzer.analyze("ok")
    strat = ResponseStrategySelector.select_strategy(intent)
    assert strat.strategy_name == "short_acknowledgment"
    assert strat.max_words <= 6


def test_strategy_venting():
    intent = SocialIntentAnalyzer.analyze("fml thak gaya")
    strat = ResponseStrategySelector.select_strategy(intent)
    assert strat.strategy_name == "supportive_response"
    assert "corporate empathy" in strat.forbidden_approaches


def test_strategy_greeting():
    intent = SocialIntentAnalyzer.analyze("kaha hai tu")
    strat = ResponseStrategySelector.select_strategy(intent)
    assert strat.strategy_name == "direct_answer"
    assert strat.max_words <= 12


def test_strategy_uses_learned_behavioral_pattern():
    intent = SocialIntentAnalyzer.analyze("kaha hai")
    pattern = {
        "response_strategy": "short_acknowledgment",
        "confidence": 0.85,
        "typical_length": "very_short",
    }
    strat = ResponseStrategySelector.select_strategy(intent, behavioral_pattern=pattern)
    assert strat.strategy_name == "short_acknowledgment"
