# ADR 0001: Local SQLite and File Storage

## Status

Accepted for first-scope app development.

## Context

The app needs to persist reusable voices and generation history in a local desktop workflow. Audio files can be large, while metadata needs to be searchable and structured. The first scope does not include cloud sync, user accounts, or collaboration.

## Decision

Use SQLite for structured metadata and the local filesystem for audio files.

Default development layout:

```text
data/app/app.sqlite3
data/app/voices/
data/app/generations/
data/app/tmp/
```

## Consequences

Benefits:

- Works offline.
- Easy to inspect during development.
- Suitable for a future packaged desktop app.
- Avoids storing large audio blobs in SQLite.
- Keeps metadata queries simple.

Tradeoffs:

- No built-in cloud sync.
- File moves outside the app can break references.
- Future packaging must define where app data lives outside source development mode.

## Alternatives Considered

### JSON Files Only

Rejected for first scope because query, migration, and consistency behavior would become fragile as history grows.

### Pluggable Database

Rejected for first scope because it adds unnecessary implementation choices before the local desktop workflow is proven.

### Store Audio in SQLite

Rejected because generated audio and reference audio are large binary assets. File storage is easier to manage, inspect, and clean up.

