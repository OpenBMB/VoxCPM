# UI Workflows

## Main Generation Workflow

1. User opens the Electron app.
2. Electron opens the React app shell and shows a loading state.
3. Electron sends AppShell status to the renderer through IPC.
4. User selects Voice Design, Voice Cloning, or Ultimate Cloning from the left sidebar.
5. The generation page renders native AppShell controls.
6. User enters target text and optional control instruction.
7. User either uploads a reference audio file or selects a saved voice after the app service exists.
8. User clicks generate.
9. The app creates a generation history record after history integration exists.
10. The AppShell calls app-layer services or backend APIs after they exist.
11. The app stores the output file and updates history.
12. User can play the result and find it later in history.

The original Gradio WebUI is started separately through `start_voxcpm.bat` or direct `app.py` commands. It is not embedded in AppShell mode.

## Shell Navigation

Left sidebar pages:

- Voice Design: default text and control-prompt generation.
- Voice Cloning: reference voice upload or saved voice reuse.
- Ultimate Cloning: reference audio plus transcript workflow.
- Voice Library: saved voices and create/import actions.
- History: generated outputs, replay, reuse, regenerate, and delete actions.
- Settings: runtime, port, device, FFmpeg, paths, logs, and cleanup actions.

The sidebar and common action icons must come from `lucide-react`. Do not add hand-drawn SVG icons for these controls.

## Save Uploaded Voice

Entry points:

- After uploading reference audio.
- After a successful generation using uploaded reference audio.

Required UI fields:

- Display name.
- Tags.
- Notes.

Success state:

- Voice appears in the Voice Library selector.
- Voice audio is copied to `data/app/voices/`.
- Metadata is saved in SQLite.

Failure state:

- Show a concise error.
- Do not create a partial active voice record.

## Select Saved Voice

UI behavior:

- Voice Library selector lists non-deleted voices.
- Selecting a voice sets its `audio_path` as the reference audio for generation.
- The user can still edit control instruction and target text.

Empty state:

- Show that no saved voices exist yet.
- Offer to upload reference audio.

## Generation History

History list item should show:

- Created time.
- Status.
- Voice display name when available.
- Short target text preview.
- Output playback control when succeeded.
- Error summary when failed.

Available actions:

- Play output audio.
- Reuse parameters.
- Regenerate.
- Delete history item.

## Reuse Parameters

When the user chooses reuse:

- Copy `input_text` into the target text field.
- Copy `control_instruction`.
- Restore selected saved voice if `voice_id` is available and not deleted.
- Restore generation parameters.
- Do not automatically start generation.

## Delete Records

Voice delete:

- Soft-delete the voice.
- Hide it from default selection lists.
- Keep historical generation records intact.

Generation delete:

- Soft-delete the generation record.
- Hide it from default history list.
- Do not physically delete output audio in the first implementation.
