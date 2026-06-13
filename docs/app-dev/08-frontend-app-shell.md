# Frontend App Shell Specification

## Purpose

The desktop app shell is the first applicationization layer for VoxCPM-Box. It should make the project feel like a local desktop product for ordinary-user dubbing workflows while preserving the existing Gradio WebUI as a separate legacy/developer route.

This is not a full model workflow rewrite. The first shell iteration provides navigation, lifecycle handling, loading states, and product structure. Native app controls are added incrementally and should connect to app adapters/services or backend APIs as those contracts are implemented.

## Product Reference

The Voicebox reference image is used as product inspiration only:

- A narrow left icon rail.
- Feature pages instead of one shared minimum UI.
- Saved voice cards for repeated use.
- A generation history list with reusable rows.
- Settings as a first-class page.

Do not copy the reference exactly. VoxCPM has its own feature map and should preserve the current generation concepts.

## Product Scenarios

VoxCPM-Box AppShell targets:

- Video spoken narration.
- AIGC short-film dubbing.
- Repeated saved-voice reuse.
- Script breakdown before generation.
- Batch task management for multi-segment scripts.
- Role profiles for characters, voice selection, notes, and prompt guidance.

## Technology

- Electron remains the desktop entry point.
- Renderer frontend uses Vite, React, and TypeScript.
- Icons must come from `lucide-react`.
- Do not add hand-drawn SVG icons for sidebar or action controls.
- The AppShell must not embed the existing Gradio UI with an iframe or webview.
- The original Gradio route remains available through `start_voxcpm.bat` and direct `app.py` commands.
- AppShell should integrate through app adapters/services rather than rewriting upstream model or Gradio source code.

## Navigation

The left sidebar must expose these pages:

- Voice Design: text and control-prompt generation without a saved reference voice.
- Voice Cloning: upload or reuse a reference voice for controllable cloning.
- Ultimate Cloning: reference audio plus transcript workflow.
- Voice Library: saved uploaded voices for repeated use.
- History: generated outputs and reusable parameters.
- Settings: runtime, device mode, port, local FFmpeg status, data path, and cleanup actions.

Future AppShell pages or sections should cover:

- Script Breakdown.
- Batch Tasks.
- Role Profiles.

Recommended `lucide-react` icons:

- `WandSparkles` for Voice Design.
- `Mic2` for Voice Cloning.
- `AudioWaveform` for Ultimate Cloning.
- `Library` for Voice Library.
- `History` for History.
- `Settings` for Settings.
- `SlidersHorizontal`, `Sparkles`, `Download`, `Star`, and `MoreHorizontal` for actions.

## Layout Rules

- Use a dense application layout, not a landing page.
- Keep the left rail narrow and persistent.
- Keep the main content area focused on the selected feature page.
- Avoid nested cards.
- Cards are allowed for individual voice assets, history rows, and settings panels.
- Use stable dimensions for the sidebar, native workbench area, voice cards, and history rows.
- Text must not overflow buttons, cards, or rows.

## Startup State

Electron should:

1. Open the app shell window.
2. Show a loading state in the renderer.
3. Notify the renderer through IPC when AppShell mode is ready or failed.
4. Start future AppShell-owned backend services only when those services exist.
5. Keep legacy Gradio startup on the original launcher route.

The renderer should not open an external browser for the main workflow.

## Generation Page Integration

First iteration:

- Show the native app shell and page controls.
- Render native AppShell generation page skeletons for Voice Design, Voice Cloning, and Ultimate Cloning.
- Use the existing Gradio UI as a reference for labels, prompts, settings, and workflow behavior, not as an embedded UI.

Later iterations:

- Move reusable controls into native React components.
- Route save-voice, history, script breakdown, batch task, and role profile actions through app adapters/services.
- Connect native generation controls to stable app APIs or service boundaries.
- Reuse upstream behavior through stable callable or launch boundaries when upstream changes are pulled.

## Voice Library Page

The first native page should support the product shape before persistence exists:

- Import Voice action.
- Create Voice action.
- Voice cards with name, short notes, tags, selected state, and action buttons.
- Future connection to `data/app/app.sqlite3` and `data/app/voices/`.

Persistence requirements are defined in `03-data-design.md`.

## History Page

The first native page should support the product shape before persistence exists:

- Row list with voice name, language/model/duration metadata, text preview, created time, favorite action, and menu action.
- Future connection to generation records and output files under `data/app/generations/`.

History requirements are defined in `01-product-prd.md` and `03-data-design.md`.

## Settings Page

Settings should show at minimum:

- Backend status.
- Backend URL and port.
- Project path.
- Output and error log paths.
- Local FFmpeg status.
- Future device mode selector.
- Future cleanup actions for temporary app files.

Settings must not mutate model internals. Device and port changes should remain app runtime configuration.

## Acceptance Criteria

- `npm.cmd run dev` opens the Electron app shell, not raw Gradio directly.
- The shell shows all six sidebar pages.
- All visible icons come from `lucide-react`.
- Generation pages do not embed Gradio through iframe or webview.
- The original Gradio route still works through `start_voxcpm.bat`.
- Renderer receives backend status through IPC.
- Closing the app stops only AppShell-owned backend services.
