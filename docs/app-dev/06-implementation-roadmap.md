# Implementation Roadmap

## Phase 1: Documentation and Conventions

Inputs:

- Current `app.py` Gradio UI.
- Current Electron shell.
- This documentation set.
- VoxCPM-Box ordinary-user scenarios: video narration, AIGC short-film dubbing, repeated voice reuse, script breakdown, batch tasks, and role profiles.

Outputs:

- Agreed data paths.
- Agreed table names and status enum.
- Implementation tasks ready for AI-assisted development.

Acceptance:

- All app docs use the same paths and table names.
- First-scope exclusions are clear.
- Upstream source preservation is documented as a default rule.

## Phase 2: Electron React App Shell

Tasks:

- Keep Electron as the desktop entry.
- Add a Vite, React, and TypeScript renderer.
- Add `lucide-react` and use it for all sidebar and common action icons.
- Add left sidebar pages for Voice Design, Voice Cloning, Ultimate Cloning, Voice Library, History, and Settings.
- Keep AppShell lifecycle in `electron/main.js`.
- Send AppShell status to the renderer through IPC.
- Render native generation pages without embedding the current Gradio WebUI.

Acceptance:

- `npm.cmd run dev` opens the app shell instead of raw Gradio directly.
- All six pages render.
- Generation pages do not contain an iframe or webview for Gradio.
- The original Gradio WebUI still starts from `start_voxcpm.bat`.
- No hand-drawn SVG icons are introduced for shell navigation or common actions.

## Phase 3: Storage Layer

Tasks:

- Create `src/voxcpm_app/`.
- Add path helpers for `data/app/`.
- Add SQLite initialization and migrations.
- Add repositories for `voices` and `generations`.
- Add file-copy and checksum helpers.

Acceptance:

- A test can create a voice record and generation record in a temporary app data root.
- Soft-delete behavior hides records from default lists.
- Audio files are copied to the expected directories.

## Phase 4: App Service Integration

Tasks:

- Add Voice Library selector to the AppShell generation UI.
- Add Save Voice action for uploaded reference audio.
- Add Generation History panel.
- Connect native AppShell controls to app-layer service contracts.
- Store successful output audio under `data/app/generations/`.

Acceptance:

- Saved voices survive app restart.
- A saved voice can be reused for generation.
- Successful and failed generations appear in history.

## Phase 5: Native Shell Feature Integration

Tasks:

- Connect the Voice Library page to app services.
- Connect the History page to app services.
- Add Settings controls for device mode, local FFmpeg status, data path, and cleanup actions.
- Gradually replace Gradio-only controls with native React controls where the app service contract is stable.
- Add product flows for video spoken narration and AIGC short-film dubbing as AppShell workflows, not model rewrites.

Acceptance:

- Saved voices can be imported, listed, selected, edited, and soft-deleted from the app shell.
- History can be viewed, replayed, reused, regenerated, and soft-deleted from the app shell.
- Settings shows runtime and local dependency status without exposing model internals.

## Phase 6: Script, Batch, and Role Workflows

Tasks:

- Add Script Breakdown as an app-layer workflow for splitting long scripts into lines, scenes, or generation tasks.
- Add Batch Task Queue as an app-layer workflow for multi-segment generation tracking.
- Add Role Profiles as app-layer metadata for character names, notes, voice selection, and prompt guidance.
- Connect these workflows to Voice Library and Generation History through app adapters/services.

Acceptance:

- A script can be split into reusable generation tasks.
- Batch tasks can track status without relying on Gradio temporary state.
- Role profiles can reuse saved voices and prompt guidance.
- No model internals or core upstream source files are rewritten for these product workflows.

## Phase 7: Tests, Upstream Sync, and Packaging Preparation

Tasks:

- Add storage-layer tests.
- Add callback-level tests for generation history lifecycle.
- Add Electron lifecycle smoke test.
- Add upstream-sync checks for the legacy/developer Gradio route and AppShell route.
- Document future packaging requirements.

Acceptance:

- Tests verify schema creation, voice save, history save, soft delete, and output path behavior.
- Upstream pull verification confirms both preserved source behavior and AppShell startup.
- Packaging notes list Python runtime, Electron files, local FFmpeg, SQLite data location, and model cache considerations.
