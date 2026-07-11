import soundfile as sf
import os
import re

# --- Configura aquí ---
SCRIPT_FILE = r"C:\Users\jonhy\Desktop\script.txt"
BLOQUES_DIR = r"C:\Users\jonhy\Desktop\bloques"
SRT_OUTPUT  = r"C:\Users\jonhy\Desktop\script_completo.srt"
MAX_CHARS   = 200
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


# Reconstruye la lista de bloques desde el texto original
with open(SCRIPT_FILE, "r", encoding="ansi") as f:
    contenido = f.read()

parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
bloques = []
for p in parrafos:
    bloques.extend(dividir_en_bloques(p))

print(f"Bloques de texto: {len(bloques)}")

# Lee la duración de cada wav existente
srt_lines = []
cursor = 0.0
encontrados = 0

for i, texto in enumerate(bloques):
    wav_path = os.path.join(BLOQUES_DIR, f"bloque_{i+1:03d}.wav")
    if not os.path.exists(wav_path):
        print(f"  ⚠ No encontrado: {wav_path} — saltando")
        continue

    info = sf.info(wav_path)
    duracion = info.duration
    inicio = segundos_a_srt(cursor)
    fin    = segundos_a_srt(cursor + duracion)
    srt_lines.append(f"{encontrados+1}\n{inicio} --> {fin}\n{texto}\n")
    cursor += duracion
    encontrados += 1

with open(SRT_OUTPUT, "w", encoding="ansi") as f:
    f.write("\n".join(srt_lines))

print(f"✓ SRT generado con {encontrados} entradas → {SRT_OUTPUT}")
