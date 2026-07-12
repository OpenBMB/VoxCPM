from voxcpm.phone_assistant import (
    _build_backend_messages,
    _build_capture_payload,
    _build_final_text,
    _extract_reply_text,
    _extract_capture_receipt,
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


def test_build_capture_payload_matches_pca_schema():
    payload = _build_capture_payload(
        user_message="Hello PCA",
        assistant_reply="Cloned reply",
        history=[("u1", "a1")],
        profile_name="reddit-female",
        backend_mode="custom",
    )
    # Required fields per schemas/pca_capture_event.schema.json.
    assert payload["source"] == "iphone_shortcut"
    assert payload["capture_type"] == "text"
    assert payload["timestamp"]
    assert payload["content"] == "Hello PCA"
    assert payload["classification"] in {"public", "internal", "confidential", "restricted"}
    assert payload["provenance"]["agent"] == "voxcpm-phone-assistant"
    assert "voxcpm-phone-assistant" in payload["tags"]
    # Off-contract keys rejected by the gateway (additionalProperties: false)
    # must not be emitted.
    assert set(payload) <= {
        "source",
        "capture_type",
        "timestamp",
        "content",
        "classification",
        "provenance",
        "tags",
    }
    assert "text" not in payload
    assert "metadata" not in payload
    assert "context_note" not in payload


def test_build_capture_payload_classification_override_validates():
    ok = _build_capture_payload(
        user_message="hi",
        assistant_reply="",
        history=[],
        profile_name="p",
        backend_mode="auto",
        classification="internal",
    )
    assert ok["classification"] == "internal"

    fallback = _build_capture_payload(
        user_message="hi",
        assistant_reply="",
        history=[],
        profile_name="p",
        backend_mode="auto",
        classification="bogus",
    )
    assert fallback["classification"] == "confidential"


def test_extract_capture_receipt_supports_capture_id():
    assert _extract_capture_receipt({"capture_id": "cap_123"}) == "cap_123"


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
