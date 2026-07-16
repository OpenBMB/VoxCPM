# Optimización de inferencia del servidor VoxCPM2

**Fecha:** 2026-07-17 · **Hardware:** RTX 3070 8GB, Windows 11, torch 2.11.0+cu128
**Restricción:** la calidad no cambia — todos los cambios activos por defecto son **bit-exact** (con la misma seed, el WAV sale byte-idéntico al código anterior).

## Resultado

Sobre el guion fijo `benchmarks/bench_script.txt` (8 bloques medidos + 2 warmup, seed 1234):

| Estado | Inferencia media/bloque | Ratio inferencia/audio |
|---|---|---|
| Código anterior (baseline) | 17.10s | 2.30× |
| **Código actual (por defecto, bit-exact)** | **15.29s** | **2.06×** |
| **Código actual + `--optimize` (triton-windows)** | **8.76s** | **1.24×** |

- Por defecto: ~11% más rápido, **bit-exact** (8/8 hashes idénticos), y sin la degradación progresiva que acababa en OOM.
- Con `--optimize` (tras instalar `triton-windows`, extra `compile` del pyproject): **43% más rápido adicional** — casi tiempo real. No es bit-exact respecto a eager (equivale a cambiar de seed: misma calidad en distribución, duraciones a <5% de eager, 0 anomalías), pero sí es **reproducible**: misma seed → mismo WAV entre pasadas y arranques compilados (verificado 8/8). Coste: el arranque tarda varios minutos compilando. `--optimize` activa solo `VOXCPM_STOP_CHECK_BATCH=8` (stop-check diferido, bit-exact) para que los CUDA graphs rindan; 0 cudagraph skips y 1 recompile en los logs; VRAM estable (reserved ~6.4 GB plano).

## Qué se cambió (por defecto, bit-exact)

1. **Tensores de posición persistentes en el bucle AR** (`voxcpm2.py`): antes se creaban 2 `torch.tensor([...], device=...)` (copias H2D bloqueantes) por patch; ahora un tensor por LM avanzado in-place con `add_(1)`. La mayor ganancia.
2. **Máscara de atención izada** (`minicpm4/model.py`): se construía con `torch.arange(8192)` por capa y por paso; ahora una vez por paso desde un buffer preallocated.
3. **Sin `zero_()` del KV cache completo** (`minicpm4/cache.py`): las posiciones rancias quedan siempre excluidas por la máscara.
4. **`torch.cuda.empty_cache()` tras cada petición** (`voxcpm_server.py`): el caching allocator retenía el pico (~7.4 de 8 GB) para siempre → spill de WDDM progresivo (2.3×→3.3× y un servidor de horas a 4.8×) y OOM tras ~15 peticiones variadas. Ahora reserved vuelve a ~5.8 GB por petición y 20+ peticiones seguidas van estables. Ver anexo en `investigacion-ram-servidor.md`.
5. **HTTP keep-alive** (`voxcpm_client.py` + servidor en HTTP/1.1): conexión persistente por hilo con reintento si quedó obsoleta.
6. **Stop-check por lotes** (`VOXCPM_STOP_CHECK_BATCH`, default 1): infraestructura para diferir el sync GPU→CPU del stop-check; bit-exact vía recorte. Con K>1 en eager pierde (~1.4% por patches descartados) — solo interesa si algún día funciona torch.compile.

## Qué se probó y se DESCARTÓ (medido, no especulado)

| Palanca | Resultado | Evidencia |
|---|---|---|
| Ventana de atención KV (slice a posiciones válidas) | ~12% MÁS LENTA (slice no contiguo → kernel SDPA peor) y no bit-exact | `bench_kvwindow.json` |
| `cudnn.benchmark` | Contraproducente: re-autotunea con cada longitud nueva del VAE | `bench_optin_flags.json` (paquete −36%) |
| TF32 | Solo tocaría el decode fp32 del VAE (~1.5% del tiempo); no compensa perder bit-exactness | perfil `VOXCPM_PROFILE` |
| `--optimize` SIN triton instalado | No compila nada ("torch.compile disabled - triton is not installed"): queda solo el warmup — minutos de arranque, ~1.5% más lento y cambia los numéricos. Con `triton-windows` instalado sí compila y gana 43% (ver arriba) | `bench_fase5_compile.json`, `bench_triton_compile.json` |

## Dónde se va el tiempo (perfil con `VOXCPM_PROFILE=1`)

Por petición (primera petición, servidor fresco): **DiT/difusión ~64%** (10-12 pasos Euler × forward 2×batch por patch), pasos LM ~27%, feat_encoder ~5%, prefill ~4%, VAE decode ~1.5%, sync del stop-check ~0.1%.

Implicación: el techo de mejora adicional estaba en la DiT — exactamente lo que torch.compile + CUDA graphs atacó al instalarse `triton-windows` (43% medido). Ganancias mayores ya exigirían tocar calidad (menos timesteps) o hardware.

## Cómo verificar / reproducir

```powershell
# Terminal A
python voxcpm_server.py            # esperar "Servidor listo"

# Terminal B — benchmark con seed fija y hashes
python benchmark_voxcpm_inference.py --script-file benchmarks\bench_script.txt --limit 8 --warmup-count 2 --seed 1234 --output-json candidato.json

# Comparar contra el baseline commiteado (tiempos + igualdad de WAVs)
python benchmark_voxcpm_inference.py --compare-json bench_fase2_k1.json candidato.json
```

- `--seed N`: el bloque i usa la seed N+i → los WAV son deterministas y comparables por SHA-256 entre runs/versiones.
- `--save-wav-dir dir`: guarda los WAV para escucharlos.
- `VOXCPM_PROFILE=1` en el servidor: imprime el desglose por buckets (CUDA events) por petición.
- Las métricas del servidor incluyen `cuda_alloc_mb` / `cuda_reserved_mb` / `vram_release_seconds` para vigilar la VRAM.

## Nota operativa

El benchmark depende de que la GPU esté libre: dos servidores cargando el modelo a la vez (o un juego/otro proceso pesado) provocan spill de VRAM y los tiempos salen inflados aunque el audio sea idéntico. Comparar siempre con servidor fresco y recién cargado.
