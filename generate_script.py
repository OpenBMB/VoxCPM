from voxcpm import VoxCPM
import soundfile as sf
import numpy as np
import os
import re

# --- Configura aquí ---
SCRIPT_FILE   = r"C:\Users\jonhy\Desktop\script.txt"
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_DIR    = r"C:\Users\jonhy\Desktop\bloques"
FINAL_OUTPUT  = r"C:\Users\jonhy\Desktop\script_completo.wav"
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


# 1. Leer y dividir el texto
with open(SCRIPT_FILE, "r", encoding="ansi") as f:
    contenido = f.read()

parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
print(f"Se encontraron {len(parrafos)} párrafos en '{SCRIPT_FILE}'")

bloques = []
for parrafo in parrafos:
    bloques.extend(dividir_en_bloques(parrafo))

print(f"Divididos en {len(bloques)} bloques de máx. {MAX_CHARS} caracteres (~20s c/u)\n")

# 2. Cargar modelo
print("Cargando modelo VoxCPM2...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)

fragmentos = []
fallidos = []

# 3. Generar cada bloque
for i, bloque in enumerate(bloques):
    bloque_path = os.path.join(OUTPUT_DIR, f"bloque_{i+1:03d}.wav")

    if os.path.exists(bloque_path):
        print(f"[{i+1}/{len(bloques)}] Ya existe: {bloque_path}")
        wav, sr = sf.read(bloque_path)
        fragmentos.append(wav)
        continue

    print(f"\n[{i+1}/{len(bloques)}] Generando ({len(bloque)} caracteres)...")
    print(f"  -> {bloque[:90]}{'...' if len(bloque) > 90 else ''}")

    try:
        wav = model.generate(
            text=bloque,
            prompt_wav_path=REFERENCE_WAV,
            reference_wav_path=REFERENCE_WAV,
            prompt_text=PROMPT_TEXT,
            cfg_value=CFG_VALUE,
            inference_timesteps=INFERENCE_TIMESTEPS,
            normalize=NORMALIZE,
            denoise=False,
        )
        sf.write(bloque_path, wav, model.tts_model.sample_rate)
        fragmentos.append(wav)
        print(f"  OK Guardado: {bloque_path}")
    except Exception as e:
        print(f"  ERROR en bloque {i+1}: {e}")
        import traceback; traceback.print_exc()
        fallidos.append(i + 1)

# 4. Concatenar en un solo archivo
if fragmentos:
    print(f"\nConcatenando {len(fragmentos)} bloques...")
    audio_completo = np.concatenate(fragmentos)
    sf.write(FINAL_OUTPUT, audio_completo, model.tts_model.sample_rate)
    print(f"OK Audio completo: {FINAL_OUTPUT}")
    print(f"\nAhora ejecuta: python generar_srt.py")

if fallidos:
    print(f"\nBloques que fallaron: {fallidos}")
