import soundfile as sf
import numpy as np
import os
import re

from voxcpm_client import DEFAULT_SERVER_URL, VoxCPMServerError, check_server, generate_wav_bytes

# --- Configura aquí ---
SCRIPT_FILE   = r"C:\Users\jonhy\Desktop\script.txt"
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_DIR    = r"C:\Users\jonhy\Desktop\bloques"
FINAL_OUTPUT  = r"C:\Users\jonhy\Desktop\script_completo.wav"
SRT_OUTPUT    = r"C:\Users\jonhy\Desktop\script_completo.srt"
MODEL_ID      = "openbmb/VoxCPM2"
SERVER_URL    = DEFAULT_SERVER_URL
MAX_CHARS     = 200  # ~20 segundos de habla por bloque
CFG_VALUE    = 2.0
INFERENCE_TIMESTEPS = 12
NORMALIZE    = False

PROMPT_TEXT = (
    "Fíjate nada más lo que acaba de pasar... porque esto que les voy a contar hoy no es un chisme cualquiera "
    "de los que se olvidan en tres días. Estamos hablando de la ruptura que todo México tenía en la boca desde "
    "el 6 de junio, sí, la de Kenia Os y Peso Pluma, pero lo que los medios no te están contando —y que nosotros "
    "encontramos después de rastrear más de doce fuentes, tres semanas de movimientos digitales y cada historia "
    "borrada— es que la verdad no está en el comunicado. La verdad estaba en el escenario, siete días antes, "
    "cuando Kenia Os se derrumbó frente a miles de personas en Monterrey cantando una canción que describe, "
    "con nombre y apellido psicológico, exactamente lo que le hicieron."
)
# ----------------------


def dividir_en_bloques(texto, max_chars=MAX_CHARS):
    frases = re.split(r'(?<=[.!?])\s+', texto.strip())
    bloques = []
    actual = ""
    for frase in frases:
        candidato = (actual + " " + frase).strip() if actual else frase
        if len(candidato) <= max_chars:
            actual = candidato
        else:
            if actual:
                bloques.append(actual)
            if len(frase) > max_chars:
                partes = re.split(r'(?<=[,;])\s+', frase)
                sub = ""
                for parte in partes:
                    c = (sub + " " + parte).strip() if sub else parte
                    if len(c) <= max_chars:
                        sub = c
                    else:
                        if sub:
                            bloques.append(sub)
                        sub = parte
                actual = sub
            else:
                actual = frase
    if actual:
        bloques.append(actual)
    return bloques


def segundos_a_srt(segundos):
    total_ms = int(round(segundos * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    sec = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def generar_srt(entradas, output_path):
    srt_lines = []
    cursor = 0.0

    for idx, (texto, duracion) in enumerate(entradas, start=1):
        inicio = segundos_a_srt(cursor)
        fin = segundos_a_srt(cursor + duracion)
        srt_lines.append(f"{idx}\n{inicio} --> {fin}\n{texto}\n")
        cursor += duracion

    with open(output_path, "w", encoding="ansi") as f:
        f.write("\n".join(srt_lines))

    return len(srt_lines)


# 1. Leer y dividir el texto
with open(SCRIPT_FILE, "r", encoding="ansi") as f:
    contenido = f.read()

parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
print(f"Se encontraron {len(parrafos)} párrafos en '{SCRIPT_FILE}'")

bloques = []
for parrafo in parrafos:
    bloques.extend(dividir_en_bloques(parrafo))

print(f"Divididos en {len(bloques)} bloques de máx. {MAX_CHARS} caracteres (~20s c/u)\n")

# 2. Comprobar servidor persistente
try:
    health = check_server(SERVER_URL)
    print(f"Servidor VoxCPM listo: {health.get('model_id', MODEL_ID)}")
except VoxCPMServerError as e:
    print(f"ERROR: {e}")
    raise SystemExit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

fragmentos = []
entradas_srt = []
fallidos = []
sample_rate = None

# 3. Generar cada bloque
for i, bloque in enumerate(bloques):
    bloque_path = os.path.join(OUTPUT_DIR, f"bloque_{i+1:03d}.wav")

    if os.path.exists(bloque_path):
        print(f"[{i+1}/{len(bloques)}] Ya existe: {bloque_path}")
        wav, sr = sf.read(bloque_path)
        sample_rate = sample_rate or sr
        fragmentos.append(wav)
        entradas_srt.append((bloque, len(wav) / sr))
        continue

    print(f"\n[{i+1}/{len(bloques)}] Generando ({len(bloque)} caracteres)...")
    print(f"  -> {bloque[:90]}{'...' if len(bloque) > 90 else ''}")

    try:
        wav_bytes = generate_wav_bytes(
            {
                "text": bloque,
                "model_id": MODEL_ID,
                "prompt_text": PROMPT_TEXT,
                "prompt_wav_path": REFERENCE_WAV,
                "reference_wav_path": REFERENCE_WAV,
                "cfg_value": CFG_VALUE,
                "inference_timesteps": INFERENCE_TIMESTEPS,
                "normalize": NORMALIZE,
                "denoise": False,
                "trim_silence_vad": False,
            },
            SERVER_URL,
        )
        with open(bloque_path, "wb") as f:
            f.write(wav_bytes)
        wav, sr = sf.read(bloque_path)
        sample_rate = sample_rate or sr
        fragmentos.append(wav)
        entradas_srt.append((bloque, len(wav) / sr))
        print(f"  OK Guardado: {bloque_path}")
    except Exception as e:
        print(f"  ERROR en bloque {i+1}: {e}")
        import traceback; traceback.print_exc()
        fallidos.append(i + 1)

# 4. Concatenar en un solo archivo
if fragmentos:
    print(f"\nConcatenando {len(fragmentos)} bloques...")
    audio_completo = np.concatenate(fragmentos)
    sf.write(FINAL_OUTPUT, audio_completo, sample_rate)
    print(f"OK Audio completo: {FINAL_OUTPUT}")
    entradas = generar_srt(entradas_srt, SRT_OUTPUT)
    print(f"OK SRT generado con {entradas} entradas: {SRT_OUTPUT}")

if fallidos:
    print(f"\nBloques que fallaron: {fallidos}")
