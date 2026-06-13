# App Architecture

## Current Architecture

The current app layer has four main parts:

- `app.py`: Gradio UI and generation orchestration.
- `electron/main.js`: desktop shell process for AppShell mode.
- `electron/renderer/`: Vite, React, and TypeScript app shell renderer.
- `run_with_local_ffmpeg.py`: local launcher wrapper that exposes project-local FFmpeg to Python subprocesses.

VoxCPM-Box treats upstream VoxCPM source as the source-development area and AppShell as the product route for ordinary-user workflows. AppShell should integrate through app adapters/services instead of rewriting core model or Gradio code.

Legacy Gradio runtime flow:

```text
start_voxcpm.bat or direct Python command
  -> starts Python backend
    -> run_with_local_ffmpeg.py
      -> app.py
        -> Gradio Blocks UI
        -> VoxCPMDemo.generate_tts_audio(...)
```

AppShell runtime flow:

```text
start_electron_shell.bat or start_electron_shell.vbs
  -> Electron main process
    -> React renderer app shell
    -> app-mode IPC state
    -> future app adapters/services
      -> upstream source behavior through stable boundaries
```

## Target First-Scope Architecture

Add a React app shell and an application layer between the UI and file/database storage.

```text
Electron main process
  -> backend lifecycle
  -> IPC backend status

React renderer app shell
  -> Voice Design page
  -> Voice Cloning page
  -> Ultimate Cloning page
  -> Voice Library page
  -> History page
  -> Settings page

Python app services
  -> app adapters/services
  -> voice library service
  -> generation history service
  -> future script breakdown service
  -> future batch task service
  -> future role profile service
  -> storage repository
    -> SQLite metadata
    -> local audio files
  -> existing generation backend
```

The model-facing generation behavior should remain isolated behind the existing `VoxCPMDemo.generate_tts_audio(...)` call.

## Proposed Module Boundaries

Recommended future modules:

```text
src/voxcpm_app/
  __init__.py
  paths.py
  db.py
  repositories.py
  voice_library.py
  generation_history.py
  audio_assets.py
  schemas.py
```

Responsibilities:

- `paths.py`: app data root and path helpers.
- `db.py`: SQLite connection, migrations, transaction helpers.
- `repositories.py`: low-level SQL operations.
- `voice_library.py`: voice save/list/update/delete workflows.
- `generation_history.py`: generation record lifecycle.
- `script_breakdown.py`: app-layer script segmentation workflows.
- `batch_tasks.py`: app-layer batch task lifecycle.
- `role_profiles.py`: role profile metadata and voice association workflows.
- `audio_assets.py`: file copy, checksum, duration, and path normalization.
- `schemas.py`: typed request/response objects or dataclasses.

## Electron Boundary

Electron remains responsible for:

- Starting and stopping future AppShell-owned backend services when they exist.
- Showing loading state while AppShell initializes.
- Sending backend status to the React renderer through IPC.
- Hosting the desktop app window.
- Closing AppShell-owned backend processes when the desktop window closes.

In default AppShell mode, Electron should not start or embed the legacy Gradio WebUI. A legacy WebUI development mode may be enabled explicitly by environment variable during development, but it is not the normal AppShell path.

Electron should not own:

- Voice Library data logic.
- Generation History data logic.
- Script breakdown, batch task, or role profile logic.
- Audio file indexing.
- Model execution behavior.

## React Renderer Boundary

The React renderer is responsible for:

- Left sidebar navigation.
- Native shell pages and layout.
- Startup, ready, failed, and exited UI states.
- Voice Library and History surfaces as app features.
- Settings display for runtime status, paths, local FFmpeg, and future cleanup actions.
- Native app-mode generation pages that can later call app services or backend APIs.
- Product pages for ordinary-user workflows such as video narration, AIGC short-film dubbing, script breakdown, batch tasks, role profiles, and saved voices.

The React renderer should use `lucide-react` for navigation and action icons. It should not introduce hand-drawn SVG icons for common shell controls.

## Legacy Gradio Boundary

Gradio remains responsible for the original developer WebUI route:

- Rendering the initial generation UI.
- Displaying the current generation controls and labels.
- Preserving the current prompt examples, advanced settings, and callback behavior.
- Passing user-selected voice paths into generation.

The AppShell must not embed Gradio as its main UI. When app-layer services exist, both AppShell and the legacy Gradio route may call those services without sharing UI code.

## Upstream Sync Boundary

Upstream source preservation means:

- Keep `app.py`, model packages, and existing CLI/WebUI behavior as close to upstream as possible.
- Put VoxCPM-Box product behavior in AppShell, app adapters/services, and app data storage.
- After pulling upstream changes, verify the legacy/developer Gradio route first, then update adapters if launch arguments or callable behavior changed.
- Avoid coupling AppShell to Gradio DOM structure or Gradio temporary files.

## Packaging Considerations

The first implementation stays in source-development mode. Future EXE packaging should include:

- Electron app files.
- Python runtime or bundled environment strategy.
- Local FFmpeg binary.
- SQLite database location policy.
- App data migration policy.
