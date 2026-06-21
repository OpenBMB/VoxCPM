from voxcpm.phone_assistant import (
    _build_backend_messages,
    _build_final_text,
    _extract_reply_text,
    _profile_path,
)


def test_build_final_text_wraps_control_instruction():
    assert _build_final_text("Hello world", "warm female voice") == "(warm female voice)Hello world"


def test_build_final_text_leaves_plain_text_when_no_control():
    assert _build_final_text("Hello world", "") == "Hello world"


def test_extract_reply_text_supports_plain_reply_field():
    assert _extract_reply_text({"reply": "  cloned voice text  "}) == "cloned voice text"


def test_extract_reply_text_supports_openai_style_response():
    payload = {"choices": [{"message": {"content": "  assistant reply  "}}]}
    assert _extract_reply_text(payload) == "assistant reply"


def test_profile_path_sanitizes_name():
    assert _profile_path("My Voice!").name == "myvoice.json"


def test_build_backend_messages_includes_context_and_history():
    messages = _build_backend_messages(
        "hello",
        [("old user", "old assistant")],
        "system context",
    )
    assert messages[0] == {"role": "system", "content": "system context"}
    assert messages[-1] == {"role": "user", "content": "hello"}
