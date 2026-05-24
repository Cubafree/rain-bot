"""Tests for LLM response parsing and validation."""
import pytest

from services.analyzer import _parse_llm_response, _verify_edge, LLMSignal


def test_parse_valid_json():
    raw = '{"our_probability": 0.75, "confidence": "high", "direction": "YES", "edge": 0.15, "reasoning": "Strong GFS signal", "data_quality": "sufficient"}'
    result = _parse_llm_response(raw, yes_price=0.60, no_price=0.40)

    assert result is not None
    assert result.direction == "YES"
    assert result.our_probability == 0.75
    assert result.confidence == "high"


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"our_probability": 0.80, "confidence": "high", "direction": "NO", "edge": 0.12, "reasoning": "Below threshold likely", "data_quality": "sufficient"}\n```'
    result = _parse_llm_response(raw, yes_price=0.70, no_price=0.30)

    assert result is not None
    assert result.direction == "NO"


def test_parse_null_direction():
    raw = '{"our_probability": 0.55, "confidence": "medium", "direction": null, "edge": 0.05, "reasoning": "Low confidence", "data_quality": "sufficient"}'
    result = _parse_llm_response(raw, yes_price=0.50, no_price=0.50)

    assert result is None


def test_parse_invalid_json():
    raw = "The market looks good. Buy YES."
    result = _parse_llm_response(raw, yes_price=0.60, no_price=0.40)

    assert result is None


def test_parse_out_of_range_probability():
    raw = '{"our_probability": 1.5, "confidence": "high", "direction": "YES", "edge": 0.15, "reasoning": "test", "data_quality": "sufficient"}'
    result = _parse_llm_response(raw, yes_price=0.60, no_price=0.40)

    assert result is None


def test_verify_edge_recomputes_mismatch():
    signal = LLMSignal(
        our_probability=0.75,
        confidence="high",
        direction="YES",
        edge=0.50,
        reasoning="test",
        data_quality="sufficient",
    )
    _verify_edge(signal, yes_price=0.60, no_price=0.40)
    # Expected edge: 0.75 - 0.60 = 0.15
    assert abs(signal.edge - 0.15) < 0.001


def test_verify_edge_no_change_when_correct():
    signal = LLMSignal(
        our_probability=0.75,
        confidence="high",
        direction="YES",
        edge=0.15,
        reasoning="test",
        data_quality="sufficient",
    )
    _verify_edge(signal, yes_price=0.60, no_price=0.40)
    assert signal.edge == 0.15
