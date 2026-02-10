"""Tests for extraction utilities."""

from rde.utils.extraction import (
    _attempt_json_repair,
    _is_valid_json,
    extract_boxed,
    extract_json_block,
    validate_trace_config,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{a{b}c}") == "a{b}c"

    def test_fraction(self):
        assert extract_boxed(r"\boxed{1/2}") == "1/2"

    def test_50_50(self):
        assert extract_boxed(r"Result: \boxed{50/50}") == "50/50"

    def test_no_box(self):
        assert extract_boxed("no boxed content here") is None

    def test_empty_box(self):
        assert extract_boxed(r"\boxed{}") == ""

    def test_multiline_before_box(self):
        text = "Line 1\nLine 2\nLine 3\n\\boxed{answer}"
        assert extract_boxed(text) == "answer"

    def test_multiple_boxes_takes_last(self):
        text = r"\boxed{first} and then \boxed{second}"
        assert extract_boxed(text) == "second"

    def test_unbalanced_braces(self):
        assert extract_boxed(r"\boxed{unclosed") is None

    def test_whitespace_stripped(self):
        assert extract_boxed(r"\boxed{  spaced  }") == "spaced"


class TestExtractJsonBlock:
    def test_raw_json(self):
        result = extract_json_block('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_markdown_code_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_markdown_no_lang(self):
        text = 'Text\n```\n{"a": 1}\n```'
        result = extract_json_block(text)
        assert result == '{"a": 1}'

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}}'
        result = extract_json_block(text)
        assert result == '{"outer": {"inner": 1}}'

    def test_json_in_prose(self):
        text = "Here is the result: {\"answer\": 42} and that's it."
        result = extract_json_block(text)
        assert result == '{"answer": 42}'

    def test_no_json(self):
        assert extract_json_block("no json here") is None

    def test_complex_arbiter_json(self):
        text = '''```json
{
  "resolution": "50/50",
  "causal_chain": "accidental opening",
  "confidence": "necessary",
  "interference_detected": [],
  "traces_adopted": ["Logician"],
  "traces_rejected": ["Believer"],
  "shadows": ["assumes purely random"]
}
```'''
        result = extract_json_block(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["resolution"] == "50/50"

    def test_json_with_trailing_comma(self):
        text = '{"key": "value", "extra": 1,}'
        result = extract_json_block(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_json_with_single_quotes(self):
        text = "{'key': 'value'}"
        result = extract_json_block(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_json_with_js_comments(self):
        text = '{"key": "value" // this is a comment\n}'
        result = extract_json_block(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_json_with_unquoted_keys(self):
        text = '{role: "Logician", system_prompt: "You are a logician."}'
        result = extract_json_block(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["role"] == "Logician"


class TestJsonRepair:
    def test_trailing_comma_object(self):
        result = _attempt_json_repair('{"a": 1, "b": 2,}')
        assert result is not None
        assert _is_valid_json(result)

    def test_trailing_comma_array(self):
        result = _attempt_json_repair('["a", "b",]')
        assert result is not None
        assert _is_valid_json(result)

    def test_single_quotes(self):
        result = _attempt_json_repair("{'key': 'value'}")
        assert result is not None
        assert _is_valid_json(result)

    def test_line_comments(self):
        result = _attempt_json_repair('{"key": "value" // comment\n}')
        assert result is not None
        assert _is_valid_json(result)

    def test_block_comments(self):
        result = _attempt_json_repair('{"key": /* inline */ "value"}')
        assert result is not None
        assert _is_valid_json(result)

    def test_unquoted_keys(self):
        result = _attempt_json_repair('{role: "test"}')
        assert result is not None
        assert _is_valid_json(result)

    def test_hopeless_input_returns_none(self):
        result = _attempt_json_repair("not json at all")
        assert result is None

    def test_valid_json_passes_through(self):
        assert _is_valid_json('{"key": "value"}')
        assert not _is_valid_json("not json")


class TestValidateTraceConfig:
    def test_valid(self):
        assert validate_trace_config({"role": "Logician", "system_prompt": "test"})

    def test_missing_role(self):
        assert not validate_trace_config({"system_prompt": "test"})

    def test_missing_system_prompt(self):
        assert not validate_trace_config({"role": "Logician"})
