# VoxCPM-Box Scope and Upstream Sync

## Purpose

VoxCPM-Box is the local desktop product layer built beside the upstream VoxCPM source. It is intended for ordinary users who need repeatable voiceover workflows, not only developers testing the model.

The upstream VoxCPM source remains the source-development area. The AppShell route is the product route.

## Target Users and Scenarios

Primary users:

- Short-video creators producing spoken video narration.
- AIGC video creators producing short-film voiceover.
- Solo creators who reuse the same character voices across many clips.
- Developers who still need the preserved legacy/developer Gradio route for source-level checks.

Primary scenarios:

- Generate narration for video spoken delivery.
- Generate dubbing for AIGC short-film scenes.
- Save uploaded voices and reuse them across sessions.
- Build role profiles that combine character metadata, voice choice, notes, and prompt guidance.
- Break a script into smaller lines or scenes before generation.
- Run batch tasks for multiple script segments.

## Product Route vs Development Route

VoxCPM-Box keeps two routes:

- AppShell route: `start_electron_shell.bat` or `start_electron_shell.vbs`.
- legacy/developer Gradio route: `start_voxcpm.bat` or direct `app.py` commands.

The AppShell route must not become a Gradio UI container. It should provide native app pages and call app adapters/services as they become available.

The legacy/developer Gradio route should remain available for upstream behavior checks, debugging, and model-source development.

## Upstream Source Preservation

Default rule:

- Do not modify core model code for AppShell features.
- Do not rewrite `app.py` just to support desktop product UI.
- Do not remove or replace the legacy/developer Gradio route.
- Prefer app adapters/services around the existing source behavior.
- Keep user data and product features in the app layer.

Acceptable integration points:

- Stable launcher boundaries.
- Python app services under an app-layer namespace.
- CLI/API/service adapters that can call upstream behavior.
- Filesystem and SQLite app data under `data/app/`.

Risky integration points:

- Editing model internals.
- Moving upstream entrypoints.
- Making AppShell depend on Gradio DOM structure.
- Treating Gradio temporary files as persistent user data.

## App-Layer Feature Direction

Committed app-layer features:

- Voice Library: saved uploaded voices for repeated use.
- Generation History: generated outputs and reusable parameters.
- Settings: device mode, local FFmpeg status, app data paths, logs, and cleanup actions.

Development features:

- Script Breakdown: split long scripts into manageable lines, scenes, or tasks.
- Batch Task Queue: generate multiple script segments and track progress/status.
- Role Profiles: store character names, notes, voice selection, and prompt guidance.
- Saved Voices: store user-uploaded voice references as reusable assets.

These features are product workflow features. They are not model training features and should not be implemented as model internals.

## Sync Strategy

When pulling new upstream VoxCPM source:

1. Preserve the upstream files and entrypoints first.
2. Verify the legacy/developer Gradio route still starts.
3. Verify the AppShell route still starts independently.
4. Update app adapters/services if upstream launch arguments or callable behavior changed.
5. Avoid patching upstream model or Gradio code unless no adapter boundary can preserve compatibility.

The intended outcome is that upstream source improvements can be reused by VoxCPM-Box without reworking the desktop product surface.

## Documentation Maintenance

Keep app-layer decisions in `docs/app-dev/`.

The root `README.md` should only contain a short pointer to VoxCPM-Box documentation. Avoid large root README rewrites because that file is likely to conflict with upstream changes.

Future implementation plans should state whether a change touches:

- upstream source behavior,
- app adapters/services,
- AppShell UI,
- app data storage,
- documentation only.

Documentation-only updates should not edit functional source files.
