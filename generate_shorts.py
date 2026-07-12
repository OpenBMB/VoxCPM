import os
import time
import traceback

import soundfile as sf

from tts_workflow import (
    audit_audio_cache,
    build_srt_entries,
    build_voxcpm_payload,
    concatenate_wavs,
    ensure_dirs,
    get_audio_info,
    parse_structured_shorts,
    prepare_cached_wav,
    read_wav,
    record_cached_wav,
    split_text_blocks,
    write_wav_bytes,
)
from voxcpm_client import DEFAULT_SERVER_URL, VoxCPMServerError, check_server, generate_wav_bytes_with_metrics

# --- Configura aqui ---
SCRIPT_FILE = r"C:\Users\jonhy\Desktop\script.txt"  # cada linea = un short
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_DIR = r"C:\Users\jonhy\Desktop\shorts"  # carpeta de salida
BLOQUES_DIR = r"C:\Users\jonhy\Desktop\shorts_bloques"  # bloques intermedios
SRT_OUTPUT = r"C:\Users\jonhy\Desktop\shorts.srt"
MODEL_ID = "openbmb/VoxCPM2"
SERVER_URL = DEFAULT_SERVER_URL
SCRIPT_ENCODING = "utf-8-sig"
SRT_ENCODING = "utf-8"

MAX_CHARS = 200  # ~20 segundos por bloque
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 12
NORMALIZE = False
AUDIT_ONLY = False
METRIC_KEYS = ("cache_seconds", "inference_seconds", "wav_write_seconds", "queue_seconds", "request_seconds")

PROMPT_TEXT = (
    "F\u00edjate nada m\u00e1s lo que acaba de pasar... porque esto que les voy a contar hoy no es un chisme cualquiera "
    "de los que se olvidan en tres d\u00edas. Estamos hablando de la ruptura que todo M\u00e9xico ten\u00eda en la boca desde "
    "el 6 de junio, s\u00ed, la de Kenia Os y Peso Pluma, pero lo que los medios no te est\u00e1n contando \u2014y que nosotros "
    "encontramos despu\u00e9s de rastrear m\u00e1s de doce fuentes, tres semanas de movimientos digitales y cada historia "
    "borrada\u2014 es que la verdad no est\u00e1 en el comunicado. La verdad estaba en el escenario, siete d\u00edas antes, "
    "cuando Kenia Os se derrumb\u00f3 frente a miles de personas en Monterrey cantando una canci\u00f3n que describe, "
    "con nombre y apellido psicol\u00f3gico, exactamente lo que le hicieron."
)
# ----------------------


def main():
    total_start = time.perf_counter()
    shorts = parse_structured_shorts(SCRIPT_FILE, encoding=SCRIPT_ENCODING)
    print(f"Se encontraron {len(shorts)} shorts en '{SCRIPT_FILE}'")

    def build_payload(block):
        return build_voxcpm_payload(
            text=block,
            model_id=MODEL_ID,
            prompt_text=PROMPT_TEXT,
            reference_wav=REFERENCE_WAV,
            cfg_value=CFG_VALUE,
            inference_timesteps=INFERENCE_TIMESTEPS,
            normalize=NORMALIZE,
        )

    def block_path_for(short_idx, block_idx):
        return os.path.join(BLOQUES_DIR, f"short_{short_idx:02d}_bloque_{block_idx:03d}.wav")

    ensure_dirs(OUTPUT_DIR, BLOQUES_DIR)

    if AUDIT_ONLY:
        total = {"total": 0, "reusable": 0, "missing": 0}
        for short_idx, short_text in enumerate(shorts, start=1):
            blocks = split_text_blocks(short_text, MAX_CHARS)
            audit = audit_audio_cache(
                blocks,
                BLOQUES_DIR,
                build_payload,
                lambda block_idx, short_idx=short_idx: block_path_for(short_idx, block_idx),
            )
            for key in total:
                total[key] += audit[key]
        print(
            "Auditoria cache: " f"{total['reusable']} reutilizables, {total['missing']} nuevos, {total['total']} total"
        )
        return

    try:
        health = check_server(SERVER_URL)
        print(f"\nServidor VoxCPM listo: {health.get('model_id', MODEL_ID)}")
        print(f"Caches de voz activas: {health.get('prompt_caches', 0)}")
    except VoxCPMServerError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1) from e

    srt_sections = []
    generated_count = 0
    reused_count = 0
    failed_count = 0
    generated_seconds = 0.0
    metric_totals = {key: 0.0 for key in METRIC_KEYS}
    metric_count = 0

    for short_idx, short_text in enumerate(shorts, start=1):
        print(f"\n{'=' * 60}")
        print(f"SHORT {short_idx}/{len(shorts)}: {short_text[:80]}{'...' if len(short_text) > 80 else ''}")
        print(f"{'=' * 60}")

        blocks = split_text_blocks(short_text, MAX_CHARS)
        print(f"  {len(blocks)} bloques de max. {MAX_CHARS} chars")

        fragments = []
        srt_entries = []
        sample_rate = None

        for block_idx, block in enumerate(blocks, start=1):
            block_path = block_path_for(short_idx, block_idx)
            payload = build_payload(block)
            cache_key, cache_source = prepare_cached_wav(BLOQUES_DIR, block_path, payload)

            if cache_source:
                info = get_audio_info(block_path)
                wav, sr = read_wav(block_path)
                record_cached_wav(BLOQUES_DIR, cache_key, payload, block_path, info)
                sample_rate = sample_rate or sr
                fragments.append(wav)
                srt_entries.append((block, info.duration))
                reused_count += 1
                print(
                    f"  [{block_idx}/{len(blocks)}] Reutilizado ({cache_source}) "
                    f"({info.duration:.2f}s), cargando..."
                )
                continue

            print(f"\n  [{block_idx}/{len(blocks)}] Generando ({len(block)} caracteres)...")
            print(f"    -> {block[:80]}{'...' if len(block) > 80 else ''}")

            try:
                request_start = time.perf_counter()
                wav_bytes, metrics = generate_wav_bytes_with_metrics(
                    payload,
                    SERVER_URL,
                )
                elapsed = time.perf_counter() - request_start
                write_wav_bytes(block_path, wav_bytes)

                info = get_audio_info(block_path)
                wav, sr = read_wav(block_path)
                sample_rate = sample_rate or sr
                fragments.append(wav)
                srt_entries.append((block, info.duration))
                record_cached_wav(BLOQUES_DIR, cache_key, payload, block_path, info, metrics=metrics)
                generated_count += 1
                generated_seconds += elapsed
                if metrics:
                    metric_count += 1
                    for key in METRIC_KEYS:
                        metric_totals[key] += float(metrics.get(key, 0.0))
                    print(
                        f"    OK Guardado: {block_path} ({info.duration:.2f}s audio, {elapsed:.2f}s local | "
                        f"cache {metrics.get('cache_seconds', 0):.2f}s, "
                        f"inferencia {metrics.get('inference_seconds', 0):.2f}s, "
                        f"wav {metrics.get('wav_write_seconds', 0):.2f}s, "
                        f"cola {metrics.get('queue_seconds', 0):.2f}s)"
                    )
                else:
                    print(f"    OK Guardado: {block_path} ({info.duration:.2f}s audio, {elapsed:.2f}s generacion)")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                failed_count += 1

        if not fragments:
            print(f"  Short {short_idx} sin fragmentos, saltando.")
            continue

        audio_short = concatenate_wavs(fragments)
        short_path = os.path.join(OUTPUT_DIR, f"short_{short_idx}.wav")
        sf.write(short_path, audio_short, sample_rate)
        print(f"\n  OK Audio guardado: {short_path}")
        srt_sections.append((f"Short {short_idx}", srt_entries))

    srt_lines = []
    for title, entries in srt_sections:
        srt_lines.append(f"#{title}\n")
        srt_lines.extend(build_srt_entries(entries))

    with open(SRT_OUTPUT, "w", encoding=SRT_ENCODING) as f:
        f.write("\n".join(srt_lines))

    total_elapsed = time.perf_counter() - total_start
    avg = generated_seconds / generated_count if generated_count else 0.0
    print(f"\nOK SRT guardado en: {SRT_OUTPUT}")
    print(f"OK Audios en: {OUTPUT_DIR}")
    print(
        "Resumen: "
        f"{generated_count} generados, {reused_count} reutilizados, "
        f"{failed_count} fallidos, {total_elapsed:.2f}s total, {avg:.2f}s promedio/bloque nuevo"
    )
    if metric_count:
        print(
            "Cuellos servidor: "
            f"cache {metric_totals['cache_seconds']:.2f}s total "
            f"({metric_totals['cache_seconds'] / metric_count:.2f}s prom), "
            f"inferencia {metric_totals['inference_seconds']:.2f}s total "
            f"({metric_totals['inference_seconds'] / metric_count:.2f}s prom), "
            f"wav {metric_totals['wav_write_seconds']:.2f}s total "
            f"({metric_totals['wav_write_seconds'] / metric_count:.2f}s prom), "
            f"cola {metric_totals['queue_seconds']:.2f}s total "
            f"({metric_totals['queue_seconds'] / metric_count:.2f}s prom), "
            f"request servidor {metric_totals['request_seconds']:.2f}s total"
        )


if __name__ == "__main__":
    main()
