import os
import time
import traceback

import soundfile as sf

from tts_workflow import (
    build_voxcpm_payload,
    concatenate_wavs,
    ensure_dirs,
    get_audio_info,
    read_paragraph_blocks,
    read_wav,
    write_srt,
    write_wav_bytes,
)
from voxcpm_client import DEFAULT_SERVER_URL, VoxCPMServerError, check_server, generate_wav_bytes_with_metrics

# --- Configura aqui ---
SCRIPT_FILE = r"C:\Users\jonhy\Desktop\script.txt"
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_DIR = r"C:\Users\jonhy\Desktop\bloques"
FINAL_OUTPUT = r"C:\Users\jonhy\Desktop\script_completo.wav"
SRT_OUTPUT = r"C:\Users\jonhy\Desktop\script_completo.srt"
MODEL_ID = "openbmb/VoxCPM2"
SERVER_URL = DEFAULT_SERVER_URL
SCRIPT_ENCODING = "utf-8-sig"
SRT_ENCODING = "utf-8"
MAX_CHARS = 200  # ~20 segundos de habla por bloque
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
    paragraphs, blocks = read_paragraph_blocks(SCRIPT_FILE, MAX_CHARS, encoding=SCRIPT_ENCODING)
    print(f"Se encontraron {len(paragraphs)} parrafos en '{SCRIPT_FILE}'")
    print(f"Divididos en {len(blocks)} bloques de max. {MAX_CHARS} caracteres (~20s c/u)\n")

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

    def block_path_for(index):
        return os.path.join(OUTPUT_DIR, f"bloque_{index:03d}.wav")

    ensure_dirs(OUTPUT_DIR)

    if AUDIT_ONLY:
        reusable = sum(1 for i in range(1, len(blocks) + 1) if os.path.exists(block_path_for(i)))
        missing = len(blocks) - reusable
        print(f"Auditoria cache: {reusable} reutilizables, {missing} nuevos, {len(blocks)} total")
        return

    try:
        health = check_server(SERVER_URL)
        print(f"Servidor VoxCPM listo: {health.get('model_id', MODEL_ID)}")
        print(f"Caches de voz activas: {health.get('prompt_caches', 0)}")
    except VoxCPMServerError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1) from e

    fragments = []
    srt_entries = []
    failed = []
    generated_count = 0
    reused_count = 0
    generated_seconds = 0.0
    metric_totals = {key: 0.0 for key in METRIC_KEYS}
    metric_count = 0
    sample_rate = None

    for i, block in enumerate(blocks, start=1):
        block_path = block_path_for(i)

        if os.path.exists(block_path):
            info = get_audio_info(block_path)
            wav, sr = read_wav(block_path)
            sample_rate = sample_rate or sr
            fragments.append(wav)
            srt_entries.append((block, info.duration))
            reused_count += 1
            print(f"[{i}/{len(blocks)}] Reutilizado: {block_path} ({info.duration:.2f}s)")
            continue

        payload = build_payload(block)
        print(f"\n[{i}/{len(blocks)}] Generando ({len(block)} caracteres)...")
        print(f"  -> {block[:90]}{'...' if len(block) > 90 else ''}")

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
            generated_count += 1
            generated_seconds += elapsed
            if metrics:
                metric_count += 1
                for key in METRIC_KEYS:
                    metric_totals[key] += float(metrics.get(key, 0.0))
                print(
                    f"  OK Guardado: {block_path} ({info.duration:.2f}s audio, {elapsed:.2f}s local | "
                    f"cache {metrics.get('cache_seconds', 0):.2f}s, "
                    f"inferencia {metrics.get('inference_seconds', 0):.2f}s, "
                    f"wav {metrics.get('wav_write_seconds', 0):.2f}s, "
                    f"cola {metrics.get('queue_seconds', 0):.2f}s)"
                )
            else:
                print(f"  OK Guardado: {block_path} ({info.duration:.2f}s audio, {elapsed:.2f}s generacion)")
        except Exception as e:
            print(f"  ERROR en bloque {i}: {e}")
            traceback.print_exc()
            failed.append(i)

    if fragments:
        print(f"\nConcatenando {len(fragments)} bloques...")
        audio_complete = concatenate_wavs(fragments)
        sf.write(FINAL_OUTPUT, audio_complete, sample_rate)
        print(f"OK Audio completo: {FINAL_OUTPUT}")
        entries = write_srt(srt_entries, SRT_OUTPUT, encoding=SRT_ENCODING)
        print(f"OK SRT generado con {entries} entradas: {SRT_OUTPUT}")

    if failed:
        print(f"\nBloques que fallaron: {failed}")

    total_elapsed = time.perf_counter() - total_start
    avg = generated_seconds / generated_count if generated_count else 0.0
    print(
        "\nResumen: "
        f"{generated_count} generados, {reused_count} reutilizados, "
        f"{len(failed)} fallidos, {total_elapsed:.2f}s total, {avg:.2f}s promedio/bloque nuevo"
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
