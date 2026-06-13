# Data Design

## Storage Strategy

Use SQLite for structured metadata and local files for audio assets.

Default development paths:

```text
data/app/app.sqlite3
data/app/voices/
data/app/generations/
data/app/tmp/
```

Paths stored in SQLite should be relative to the project root where practical.

## Tables

### `voices`

Stores reusable reference voices.

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | UUID string |
| `display_name` | TEXT NOT NULL | User-facing name |
| `tags` | TEXT NOT NULL DEFAULT `[]` | JSON array of strings |
| `notes` | TEXT NOT NULL DEFAULT `` | User notes |
| `source` | TEXT NOT NULL DEFAULT `upload` | `upload`, `microphone`, `imported`, `unknown` |
| `audio_path` | TEXT NOT NULL | Relative path to stored audio |
| `audio_sha256` | TEXT NOT NULL | File checksum |
| `duration_seconds` | REAL | Nullable when unknown |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC |
| `last_used_at` | TEXT | ISO 8601 UTC |
| `deleted_at` | TEXT | Soft delete timestamp |

### `generations`

Stores generation attempts and outputs.

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | UUID string |
| `input_text` | TEXT NOT NULL | Target text |
| `control_instruction` | TEXT NOT NULL DEFAULT `` | Control prompt |
| `voice_id` | TEXT | Nullable FK-like reference to `voices.id` |
| `reference_audio_path` | TEXT | Stored or uploaded reference path |
| `prompt_text` | TEXT NOT NULL DEFAULT `` | Ultimate cloning transcript |
| `cfg_value` | REAL NOT NULL | Generation parameter |
| `inference_timesteps` | INTEGER NOT NULL | Generation parameter |
| `normalize` | INTEGER NOT NULL | Boolean 0/1 |
| `denoise` | INTEGER NOT NULL | Boolean 0/1 |
| `output_audio_path` | TEXT | Relative path after success |
| `sample_rate` | INTEGER | Output sample rate |
| `status` | TEXT NOT NULL | See status enum |
| `error_summary` | TEXT NOT NULL DEFAULT `` | Short failure message |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC |
| `deleted_at` | TEXT | Soft delete timestamp |

Generation status enum:

```text
pending
running
succeeded
failed
cancelled
deleted
```

## File Naming

Recommended voice file path:

```text
data/app/voices/{voice_id}{original_extension}
```

Recommended generation output path:

```text
data/app/generations/{generation_id}.wav
```

Temporary app files:

```text
data/app/tmp/{uuid}-{safe_filename}
```

## Deletion Policy

First implementation uses soft delete:

- Set `deleted_at`.
- Hide records with `deleted_at IS NOT NULL` from default lists.
- Do not immediately remove files.

Physical cleanup can be implemented later as a separate maintenance action.

## Migration Policy

Use numbered migrations or an explicit `schema_version` table.

Minimum required table:

```text
schema_version
  version INTEGER NOT NULL
  applied_at TEXT NOT NULL
```

