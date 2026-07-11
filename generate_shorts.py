from voxcpm import VoxCPM
import soundfile as sf
import numpy as np
import os
import re

# --- Configura aquí ---
SCRIPT_FILE   = r"C:\Users\jonhy\Desktop\script.txt"       # cada línea = un short
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_DIR    = r"C:\Users\jonhy\Desktop\shorts"            # carpeta de salida
BLOQUES_DIR   = r"C:\Users\jonhy\Desktop\shorts_bloques"    # bloques intermedios
SRT_OUTPUT    = r"C:\Users\jonhy\Desktop\shorts.srt"

MAX_CHARS = 200  # ~20 segundos por bloque

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


def segundos_a_srt(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


# Lee el archivo: formato estructurado con SHORT N / Título / Descripción / Script:
def parsear_shorts(ruta):
    with open(ruta, "r", encoding="ansi") as f:
        contenido = f.read()

    # Divide por separadores --- o por **SHORT N**
    bloques = re.split(r'^---\s*$', contenido, flags=re.MULTILINE)
    shorts = []
    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue
        # Extrae el texto en la misma línea que "Script:"
        match = re.search(r'^Script:\s*(.+)', bloque, re.MULTILINE)
        if match:
            texto = match.group(1).strip()
            if texto:
                shorts.append(texto)
    return shorts

shorts = parsear_shorts(SCRIPT_FILE)
print(f"Se encontraron {len(shorts)} shorts en '{SCRIPT_FILE}'")

# Carga el modelo una sola vez
print("\nCargando modelo VoxCPM2...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BLOQUES_DIR, exist_ok=True)

sr = model.tts_model.sample_rate
srt_sections = []  # lista de (titulo, lista de (texto, duracion))

for short_idx, texto_short in enumerate(shorts, start=1):
    print(f"\n{'='*60}")
    print(f"SHORT {short_idx}/{len(shorts)}: {texto_short[:80]}{'...' if len(texto_short) > 80 else ''}")
    print(f"{'='*60}")

    bloques = dividir_en_bloques(texto_short)
    print(f"  {len(bloques)} bloques de máx. {MAX_CHARS} chars")

    fragmentos = []
    textos_bloque = []

    for b_idx, bloque in enumerate(bloques, start=1):
        bloque_path = os.path.join(BLOQUES_DIR, f"short_{short_idx:02d}_bloque_{b_idx:03d}.wav")

        if os.path.exists(bloque_path):
            print(f"  [{b_idx}/{len(bloques)}] Ya existe, cargando...")
            wav, _ = sf.read(bloque_path)
            fragmentos.append(wav)
            textos_bloque.append(bloque)
            continue

        print(f"\n  [{b_idx}/{len(bloques)}] Generando ({len(bloque)} caracteres)...")
        print(f"    → {bloque[:80]}{'...' if len(bloque) > 80 else ''}")

        try:
            wav = model.generate(
                text=bloque,
                prompt_wav_path=REFERENCE_WAV,
                reference_wav_path=REFERENCE_WAV,
                prompt_text=PROMPT_TEXT,
                cfg_value=2.0,
                inference_timesteps=12,
                normalize=False,
                denoise=False,
            )
            sf.write(bloque_path, wav, sr)
            fragmentos.append(wav)
            textos_bloque.append(bloque)
            print(f"    ✓ Guardado: {bloque_path}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback; traceback.print_exc()

    if not fragmentos:
        print(f"  ⚠ Short {short_idx} sin fragmentos, saltando.")
        continue

    # Concatena los bloques del short en un solo audio
    audio_short = np.concatenate(fragmentos)
    short_path = os.path.join(OUTPUT_DIR, f"short_{short_idx}.wav")
    sf.write(short_path, audio_short, sr)
    print(f"\n  ✓ Audio guardado: {short_path}")

    # Acumula info para el SRT
    srt_sections.append((f"Short {short_idx}", textos_bloque, fragmentos))

# Genera el SRT unificado
srt_lines = []
for titulo, textos, wavs in srt_sections:
    srt_lines.append(f"#{titulo}\n")
    cursor = 0.0
    for idx, (texto, wav) in enumerate(zip(textos, wavs), start=1):
        duracion = len(wav) / sr
        inicio = segundos_a_srt(cursor)
        fin    = segundos_a_srt(cursor + duracion)
        srt_lines.append(f"{idx}\n{inicio} --> {fin}\n{texto}\n")
        cursor += duracion

with open(SRT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))

print(f"\n✓ SRT guardado en: {SRT_OUTPUT}")
print(f"✓ Audios en: {OUTPUT_DIR}")
