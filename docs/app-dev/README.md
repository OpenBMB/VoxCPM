# VoxCPM App Development Docs

This documentation describes VoxCPM-Box application-layer development on top of the current VoxCPM project. It intentionally avoids model internals and focuses on desktop app behavior, local data, user workflows, app adapters/services, and implementation boundaries.

## Current App-Layer State

- The main generation UI is implemented with Gradio Blocks in `app.py`.
- The original Gradio route remains available through `start_voxcpm.bat` and direct `app.py` commands.
- The Electron AppShell route is a separate React app mode launched through `start_electron_shell.bat` or `start_electron_shell.vbs`.
- `run_with_local_ffmpeg.py` prepares project-local FFmpeg access before launching Python entrypoints.
- The current Electron renderer uses Vite, React, TypeScript, and `lucide-react`.
- There is no persistent Voice Library, Generation History, app data directory, or standalone application API layer yet.
- VoxCPM-Box targets ordinary-user voiceover workflows while preserving upstream source behavior.

## First Scope

The first app-development scope is:

- Voice Library: save uploaded reference voices for reuse.
- Generation History: record generated outputs and their reusable parameters.
- Local app data management: store structured metadata in SQLite and audio files on disk.

Product scenarios tracked by VoxCPM-Box:

- Video spoken narration.
- AIGC short-film dubbing.
- Repeated voice reuse.
- Script breakdown and batch task workflows.
- Role profiles and saved voices.

Out of scope for the first implementation:

- Cloud sync.
- User accounts.
- Model training or model-level voice tuning.
- Multi-device collaboration.
- Removing the original Gradio developer route.
- Hand-drawn sidebar or action SVG icons.

## Document Index

- [01 Product PRD](01-product-prd.md)
- [02 Architecture](02-architecture.md)
- [03 Data Design](03-data-design.md)
- [04 API Contracts](04-api-contracts.md)
- [05 UI Workflows](05-ui-workflows.md)
- [06 Implementation Roadmap](06-implementation-roadmap.md)
- [07 Test Acceptance](07-test-acceptance.md)
- [08 Frontend App Shell](08-frontend-app-shell.md)
- [09 VoxCPM-Box Scope and Upstream Sync](09-voxcpm-box-scope-and-upstream-sync.md)
- [ADR 0001: Local SQLite and File Storage](adr/0001-local-sqlite-and-file-storage.md)

## Default App Data Layout

Development data root:

```text
F:\.VoxCPM\VoxCPM\data\app\
```

Relative project paths:

```text
data/app/app.sqlite3
data/app/voices/
data/app/generations/
data/app/tmp/
```

The SQLite database stores metadata. Audio files stay on disk and are referenced by relative paths and checksums.

## Roadmap Overview

1. Phase 1: Documentation and app data conventions.
2. Phase 2: Storage layer and SQLite schema.
3. Phase 3: React app shell with Voicebox-style navigation and native app-mode pages.
4. Phase 4: App service integration for Voice Library and History.
5. Phase 5: Tests, migration checks, and packaging preparation.
