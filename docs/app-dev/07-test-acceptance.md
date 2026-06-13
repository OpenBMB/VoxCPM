# Test and Acceptance Plan

## Storage Tests

Required scenarios:

- Initialize a new SQLite database.
- Apply schema migrations once.
- Apply schema migrations repeatedly without duplicating schema.
- Create, list, update, and soft-delete a voice.
- Create, update, and soft-delete a generation.
- Verify deleted records are hidden by default.
- Verify audio files are copied to `data/app/voices/` and `data/app/generations/`.

## UI Workflow Tests

Required scenarios:

- Voice Library empty state appears when no saved voices exist.
- Uploaded reference audio can be saved as a voice.
- Saved voice appears in selector after save.
- Selecting a saved voice uses its stored audio path.
- Successful generation creates a `succeeded` history record.
- Failed generation creates a `failed` history record with `error_summary`.
- Reuse action restores generation parameters without auto-generating.

## Electron Lifecycle Tests

Required scenarios:

- Electron starts the Python backend.
- Electron loads `http://127.0.0.1:8808` inside the app window.
- Closing the Electron window stops backend processes.
- Port `8808` is not left in `Listen` state after close.

## Regression Tests

Existing generation behavior must remain valid:

- Voice design with no reference audio still works.
- Reference-audio generation still accepts an uploaded audio path.
- Ultimate cloning still passes `prompt_wav_path` and `prompt_text`.
- Local FFmpeg wrapper remains available for formats such as `.m4a`.

## Documentation Acceptance

Docs are acceptable when:

- Every first-scope feature appears in PRD, architecture, data design, UI workflows, and tests.
- Table names, field names, status enum, and paths are consistent.
- Model internals and training are not documented as app-layer scope.
- A future AI agent can implement Phase 2 without asking for storage path, table purpose, or deletion policy.

