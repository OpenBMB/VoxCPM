from voxcpm import VoxCPM
import soundfile as sf

# --- Configura aquí ---
REFERENCE_WAV  = r"C:\Users\jonhy\Desktop\audio-40s.wav"
OUTPUT_PATH    = r"C:\Users\jonhy\Desktop\clonado_ultimate_d_output.wav"
TEXT           = "¿Qué pasó exactamente en esos meses entre la grabación del video y la primera aparición pública de la nueva pareja? Eso, fíjense, es exactamente lo que nadie ha podido confirmar con declaraciones directas."
PROMPT_TEXT    = (
    "Fíjate nada más lo que acaba de pasar... porque esto que les voy a contar hoy no es un chisme cualquiera "
    "de los que se olvidan en tres días. Estamos hablando de la ruptura que todo México tenía en la boca desde "
    "el 6 de junio, sí, la de Kenia Os y Peso Pluma, pero lo que los medios no te están contando —y que nosotros "
    "encontramos después de rastrear más de doce fuentes, tres semanas de movimientos digitales y cada historia "
    "borrada— es que la verdad no está en el comunicado. La verdad estaba en el escenario, siete días antes, "
    "cuando Kenia Os se derrumbó frente a miles de personas en Monterrey cantando una canción que describe, "
    "con nombre y apellido psicológico, exactamente lo que le hicieron."
)
# ----------------------

print("Cargando modelo VoxCPM2...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=True)

print("Generando audio clonado (Ultimate Cloning)...")
wav = model.generate(
    text=TEXT,
    prompt_wav_path=REFERENCE_WAV,
    reference_wav_path=REFERENCE_WAV,
    prompt_text=PROMPT_TEXT,
    cfg_value=2.0,           
    # escala de guía: qué tan fuerte sigue el texto. Rango recomendado: 1.5–3.0 
    # 1.0	Sin guía extra — el modelo es más "creativo", puede sonar más natural pero menos fiel
    # 2.0 (default)	Balance entre naturalidad y fidelidad al texto/voz
    # 3.0+	Muy apegado al texto y referencia, pero puede sonar robótico o exagerado
    inference_timesteps=10,  
    # pasos de difusión: más pasos = mejor calidad pero más lento. Rango: 1–100
    normalize=True,         
    # normaliza el texto antes de sintetizar (útil si hay números o símbolos raros)
    denoise=False,           
    # doenisea el audio de REFERENCIA antes de procesarlo (útil si tiene ruido de fondo)
)

sf.write(OUTPUT_PATH, wav, model.tts_model.sample_rate)
print(f"Audio guardado en: {OUTPUT_PATH}")

# python clone_test.py