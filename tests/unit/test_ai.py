"""Unit tests for AI provider abstraction, router, and response validator."""

import pytest
from pydantic import BaseModel

from app.ai.provider import LLMResponse
from app.ai.router import MockLLMProvider, get_llm_provider
from app.ai.validator import ResponseValidator


class SampleSchema(BaseModel):
    intent: str
    reply_text: str
    internal_reasoning: str
    humor_style_applied: str


@pytest.mark.asyncio
async def test_mock_llm_provider_generation():
    provider = MockLLMProvider(canned_response="Surviving on 128MB RAM and digital irony.")
    resp = await provider.generate(prompt="How are you?")
    assert isinstance(resp, LLMResponse)
    assert resp.content == "Surviving on 128MB RAM and digital irony."
    assert resp.provider == "mock"
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_mock_llm_provider_structured_generation():
    provider = MockLLMProvider()
    resp = await provider.generate(
        prompt="Tell a joke",
        response_schema=SampleSchema,
    )
    assert resp.structured_data is not None
    assert resp.structured_data["intent"] == "REPLY"
    assert "reply_text" in resp.structured_data


def test_router_mock_resolution():
    provider = get_llm_provider(force_mock=True)
    assert isinstance(provider, MockLLMProvider)


def test_response_validator_valid_case():
    validator = ResponseValidator(max_length=400)
    res = validator.validate("That sounds completely absurd, but I admire the dedication.")
    assert res.is_valid
    assert res.rejection_reason is None
    assert res.sanitized_text == "that sounds completely absurd, but i admire the dedication"


def test_response_validator_empty():
    validator = ResponseValidator()
    res = validator.validate("   ")
    assert not res.is_valid
    assert res.violated_rule == "empty_response"


def test_response_validator_max_length():
    validator = ResponseValidator(max_length=50)
    res = validator.validate("This sentence is definitely longer than fifty characters in total length.")
    assert not res.is_valid
    assert res.violated_rule == "max_length"


def test_response_validator_dangerous_content():
    validator = ResponseValidator()
    res = validator.validate("Here are instructions on how to build a bomb for fun.")
    assert not res.is_valid
    assert res.violated_rule == "safety_violation"


def test_response_validator_prompt_leakage():
    validator = ResponseValidator()
    res = validator.validate("I am an AI and [SYSTEM] persona directive is to act as Vesper.")
    assert not res.is_valid
    assert res.violated_rule == "prompt_leakage"


def test_response_validator_raw_json_or_code_fences():
    validator = ResponseValidator()
    res1 = validator.validate('{"intent": "REPLY", "reply_text": "hello"}')
    assert not res1.is_valid
    assert res1.violated_rule == "unwanted_metadata"

    res2 = validator.validate("```json\n{'reply': 'hi'}\n```")
    assert not res2.is_valid
    assert res2.violated_rule == "unwanted_metadata"


def test_response_validator_repetition():
    validator = ResponseValidator(repetition_similarity_threshold=0.85)
    previous_replies = ["Classic human move: panic first, ask questions later."]
    candidate = "Classic human move: panic first, ask question later."  # Near duplicate

    res = validator.validate(candidate, recent_replies=previous_replies)
    assert not res.is_valid
    assert res.violated_rule == "repetition"


def test_groq_provider_initialization():
    from app.ai.groq import GroqProvider

    provider = GroqProvider(api_key="gsk_dummy_test_key", model="openai/gpt-oss-120b")
    assert provider.api_key == "gsk_dummy_test_key"
    assert provider.model == "openai/gpt-oss-120b"


def test_openrouter_provider_initialization():
    from app.ai.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(api_key="sk-or-dummy", model="nvidia/nemotron-3-super-120b-a12b:free")
    assert provider.api_key == "sk-or-dummy"
    assert provider.model == "nvidia/nemotron-3-super-120b-a12b:free"


