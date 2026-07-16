# Investigación: RAM del servidor VoxCPM al 80% en la rama `performance`

**Fecha:** 2026-07-16
**Síntoma:** al arrancar el servidor (`.\.venv\Scripts\python.exe .\voxcpm_server.py`) en la rama `performance`, la RAM del PC subía a ~80%, mientras que en `main` (venv `venv_voxcpm`) se quedaba en ~35%.

## Conclusión

La causa **no era el código** (ni las métricas nuevas del servidor, ni diferencias en la carga del modelo — el código de `src/voxcpm` es idéntico entre ramas). Era la **versión de torch de cada venv**: en Windows, torch ≥ 2.7 usa **mimalloc** como allocator de CPU, y mimalloc retiene ~11 GB de memoria comprometida tras la carga del modelo en vez de devolverla al sistema.

**Fix aplicado** (commit `4b2152f`): poner `MIMALLOC_PURGE_DELAY=0` antes de que se cargue torch, en `voxcpm_server.py`:

```python
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
```

## Entornos comparados

| | main | performance |
|---|---|---|
| Venv | `venv_voxcpm` (Python 3.11.9) | `.venv` (uv, Python 3.11.15) |
| torch | 2.6.0+cu124 | 2.11.0+cu128 |
| transformers | 4.51.3 | 5.3.0 |
| CUDA disponible | Sí (RTX 3070) | Sí (RTX 3070) |
| VRAM del modelo | 5.3 GB | 5.3 GB |

## Mediciones (memoria comprometida del proceso, VoxCPM2 cargado, `optimize=False`)

| Configuración | Commit RAM |
|---|---|
| main, torch 2.6.0+cu124 | 8.5 GB |
| performance, torch 2.11.0+cu128 (sin fix) | **18.9 GB** |
| performance + `MIMALLOC_PURGE_DELAY=0` | **7.4 GB** ✅ |
| performance + construcción en bf16 (alternativa no aplicada) | 7.7 GB |

Otras mediciones que acotaron la causa:

- Solo `import torch` + init de CUDA: main 1.37 GB vs performance 1.65 GB — la runtime de CUDA 12.8 solo explica ~0.3 GB.
- El pico de commit durante la carga es similar en ambos (21.4 GB vs 24.8 GB); la diferencia es que main **devuelve** la memoria al terminar y performance (sin fix) **la retiene**.
- No es memoria pinned de CUDA: `torch.cuda.host_memory_stats()` estaba a cero y `torch.cuda.empty_cache()` no liberaba nada.

## Mecanismo

Durante `VoxCPM2Model.from_local` el modelo se construye en **fp32 en CPU** (~10.6 GB), se convierte a **bf16** (~5.3 GB) y se mueve a la GPU. Todo ese churn de CPU se libera lógicamente, pero el mimalloc que traen los wheels de Windows de torch ≥ 2.7 mantiene las páginas comprometidas en su heap durante toda la vida del proceso. `MIMALLOC_PURGE_DELAY=0` le ordena descomprometer las páginas liberadas inmediatamente, sin coste apreciable (la inferencia es GPU-bound).

## Verificación del fix

- `pytest tests/test_voxcpm_server.py`: 3 tests pasan (incluido `test_mimalloc_purge_delay_configured`).
- Servidor real arrancado con el fix: commit de 7.4 GB tras "Modelo cargado" (antes 18.9 GB).
- Generación por HTTP OK: `cache_hit: true` en la segunda petición, `inference_seconds: 8.16` para 2.7 s de audio (servidor frío, sin `--optimize`) — sin regresión de velocidad.

## Nota para el futuro

Cualquier punto de entrada nuevo que cargue el modelo directamente (por ejemplo `app.py`, scripts de benchmark standalone, scripts de fine-tuning) necesita la misma variable de entorno antes de importar torch. Alternativa de código (no aplicada, validada): construir el modelo directamente en bf16 para reducir también el pico de carga.
