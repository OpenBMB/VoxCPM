# VoxCPM Desktop App PRD

## 1. Document Information

- Version: 0.1
- Status: Draft
- Scope: Application layer only
- First implementation target: Electron app shell, Voice Library, and Generation History

## 2. Background and Goals

VoxCPM currently provides a local Gradio WebUI for speech generation. Users can upload reference audio and generate speech, but the app does not persist uploaded voices or generation results as reusable assets.

The goal is to evolve the project into a local desktop app workflow where users can save reference voices, reuse them across generations, and review past outputs without manually managing temporary Gradio files.

The first desktop experience uses a Voicebox-inspired shell: a left icon sidebar, feature pages for the existing VoxCPM generation modes, native Voice Library and History surfaces, and Settings. The shell is a separate app mode. The original Gradio WebUI remains available through the existing startup route for development and regression checks.

## 3. Users and Scenarios

Target users:

- A solo creator repeatedly generating speech with a small set of reference voices.
- A developer testing prompts and generation parameters across multiple attempts.
- A local-first desktop user who does not want cloud accounts or remote storage.

Core scenarios:

- Save an uploaded reference audio clip as a named reusable voice.
- Select a saved voice when generating new audio.
- Review previous generations and replay generated outputs.
- Reuse a historical generation as a starting point for a new generation.
- Switch between VoxCPM feature pages without treating all modes as one undifferentiated WebUI.

## 4. App Shell Requirements

The Electron app shell must expose these pages:

- Voice Design.
- Voice Cloning.
- Ultimate Cloning.
- Voice Library.
- History.
- Settings.

Required behavior:

- Start in a desktop window with a loading state while the backend starts.
- Start the current backend through `run_with_local_ffmpeg.py app.py --port 8808 --device cuda`.
- Keep the backend lifecycle in Electron.
- Render native AppShell pages for generation modes without embedding Gradio.
- Preserve the original Gradio startup route as the legacy/developer WebUI.
- Use `lucide-react` for sidebar and action icons.
- Do not use hand-drawn SVG icons for the shell navigation or common actions.

Out of scope:

- Removing the original Gradio WebUI.
- Replacing the generation callback before app-layer services exist.

## 5. First-Scope Features

### 5.1 Voice Library

Users can save a reference audio file as a voice asset.

Required metadata:

- `display_name`
- `tags`
- `notes`
- `source`
- `audio_path`
- `audio_sha256`
- `duration_seconds`
- `created_at`
- `updated_at`
- `last_used_at`
- `deleted_at`

Required behavior:

- Save an uploaded reference audio file into `data/app/voices/`.
- Store metadata in SQLite.
- List saved voices in the generation UI.
- Select a saved voice as the `reference_wav_path` for generation.
- Edit voice display name, tags, and notes.
- Soft-delete voices by setting `deleted_at`.

Out of scope:

- Cloud sync.
- User accounts.
- Voice training.
- Voice embedding search.
- Multi-speaker project management.

### 5.2 Generation History

Users can view generated outputs and reuse generation parameters.

Required metadata:

- `input_text`
- `control_instruction`
- `voice_id`
- `reference_audio_path`
- `prompt_text`
- `cfg_value`
- `inference_timesteps`
- `normalize`
- `denoise`
- `output_audio_path`
- `sample_rate`
- `status`
- `error_summary`
- `created_at`
- `updated_at`
- `deleted_at`

Required behavior:

- Create a `generation` record when generation starts.
- Update the record to `succeeded` with output path and sample rate when generation finishes.
- Update the record to `failed` with an error summary when generation fails.
- List recent generations.
- Play historical generated audio.
- Copy parameters from a history item into the generation form.
- Regenerate from a history item.
- Soft-delete history items.

Status enum:

```text
pending
running
succeeded
failed
cancelled
deleted
```

### 5.3 Local File Management

Required behavior:

- Store uploaded voice files under `data/app/voices/`.
- Store generated output files under `data/app/generations/`.
- Store temporary app-managed files under `data/app/tmp/`.
- Never rely on Gradio temp paths as long-term storage.
- Prefer relative paths from the project root in SQLite.

## 6. Non-Functional Requirements

- Local-first: no network service is required for app data.
- Recoverable: metadata and files survive app restarts.
- Traceable: every generation record keeps enough parameters to reproduce a similar run.
- Conservative deletion: first implementation uses soft delete.
- AI-friendly implementation: storage paths, schema, and statuses are explicitly documented.
- App-like desktop workflow: the default desktop entry should be the Electron app shell, not a raw browser tab.

## 7. Acceptance Criteria

- The Electron app opens a shell with sidebar pages for Voice Design, Voice Cloning, Ultimate Cloning, Voice Library, History, and Settings.
- The shell uses `lucide-react` icons only for navigation and common actions.
- Generation pages render in AppShell mode without an embedded Gradio iframe or webview.
- The original Gradio route can still be started separately from `start_voxcpm.bat`.
- A user can save a reference audio file as a named voice and reuse it after restarting the app.
- A user can generate speech with a saved voice.
- A successful generation appears in history with playable output audio.
- A failed generation appears in history with an error summary.
- A history item can repopulate the generation form.
- Soft-deleted voices and history items are hidden from default lists.
- No first-scope feature requires model internals, cloud sync, or user accounts.
