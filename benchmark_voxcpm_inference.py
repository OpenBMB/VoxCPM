import argparse
import json
import statistics
import time
from pathlib import Path

from tts_workflow import build_voxcpm_payload, read_paragraph_blocks
from voxcpm_client import DEFAULT_SERVER_URL, VoxCPMServerError, check_server, generate_wav_bytes_with_metrics

SCRIPT_FILE = r"C:\Users\jonhy\Desktop\script.txt"
REFERENCE_WAV = r"C:\Users\jonhy\Desktop\audio-40s.wav"
MODEL_ID = "openbmb/VoxCPM2"
SERVER_URL = DEFAULT_SERVER_URL
MAX_CHARS = 200
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 12
NORMALIZE = False
DEFAULT_BLOCK_LIMIT = 8
DEFAULT_AUDIO_DIFF_THRESHOLD = 0.30

PROMPT_TEXT = (
    "F\u00edjate nada m\u00e1s lo que acaba de pasar... porque esto que les voy a contar hoy no es un chisme cualquiera "
    "de los que se olvidan en tres d\u00edas. Estamos hablando de la ruptura que todo M\u00e9xico ten\u00eda en la boca desde "
    "el 6 de junio, s\u00ed, la de Kenia Os y Peso Pluma, pero lo que los medios no te est\u00e1n contando \u2014y que nosotros "
    "encontramos despu\u00e9s de rastrear m\u00e1s de doce fuentes, tres semanas de movimientos digitales y cada historia "
    "borrada\u2014 es que la verdad no est\u00e1 en el comunicado. La verdad estaba en el escenario, siete d\u00edas antes, "
    "cuando Kenia Os se derrumb\u00f3 frente a miles de personas en Monterrey cantando una canci\u00f3n que describe, "
    "con nombre y apellido psicol\u00f3gico, exactamente lo que le hicieron."
)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_ratio(metrics):
    audio_seconds = safe_float(metrics.get("audio_seconds"))
    inference_seconds = safe_float(metrics.get("inference_seconds"))
    if audio_seconds <= 0:
        return 0.0
    return inference_seconds / audio_seconds


def summarize_metrics(results):
    completed = [
        result for result in results if result.get("ok") and result.get("metrics") and not result.get("warmup")
    ]
    failed = [result for result in results if not result.get("ok")]
    warmups = [result for result in results if result.get("warmup")]
    if not completed:
        return {
            "completed": 0,
            "failed": len(failed),
            "warmups": len(warmups),
            "avg_inference_seconds": 0.0,
            "avg_audio_seconds": 0.0,
            "avg_ratio": 0.0,
            "median_ratio": 0.0,
            "total_cache_seconds": 0.0,
            "total_request_seconds": 0.0,
            "total_inference_seconds": 0.0,
        }

    ratios = [metric_ratio(result["metrics"]) for result in completed]
    inference_seconds = [safe_float(result["metrics"].get("inference_seconds")) for result in completed]
    audio_seconds = [safe_float(result["metrics"].get("audio_seconds")) for result in completed]
    request_seconds = [safe_float(result["metrics"].get("request_seconds")) for result in completed]
    cache_seconds = [safe_float(result["metrics"].get("cache_seconds")) for result in completed]
    return {
        "completed": len(completed),
        "failed": len(failed),
        "warmups": len(warmups),
        "avg_inference_seconds": statistics.mean(inference_seconds),
        "avg_audio_seconds": statistics.mean(audio_seconds),
        "avg_ratio": statistics.mean(ratios),
        "median_ratio": statistics.median(ratios),
        "total_cache_seconds": sum(cache_seconds),
        "total_request_seconds": sum(request_seconds),
        "total_inference_seconds": sum(inference_seconds),
    }


def compare_summaries(baseline, candidate):
    baseline_ratio = safe_float(baseline.get("median_ratio"))
    candidate_ratio = safe_float(candidate.get("median_ratio"))
    baseline_avg_ratio = safe_float(baseline.get("avg_ratio"))
    candidate_avg_ratio = safe_float(candidate.get("avg_ratio"))
    baseline_inference = safe_float(baseline.get("avg_inference_seconds"))
    candidate_inference = safe_float(candidate.get("avg_inference_seconds"))
    ratio_improvement = 0.0
    inference_improvement = 0.0
    if baseline_ratio > 0:
        ratio_improvement = ((baseline_ratio - candidate_ratio) / baseline_ratio) * 100
    if baseline_inference > 0:
        inference_improvement = ((baseline_inference - candidate_inference) / baseline_inference) * 100
    return {
        "baseline_median_ratio": baseline_ratio,
        "candidate_median_ratio": candidate_ratio,
        "median_ratio_improvement_percent": ratio_improvement,
        "baseline_avg_ratio": baseline_avg_ratio,
        "candidate_avg_ratio": candidate_avg_ratio,
        "baseline_avg_inference_seconds": baseline_inference,
        "candidate_avg_inference_seconds": candidate_inference,
        "inference_improvement_percent": inference_improvement,
    }


def _results_by_index(results):
    return {
        result.get("index"): result
        for result in results
        if result.get("ok") and result.get("metrics") and not result.get("warmup")
    }


def find_audio_anomalies(baseline_results, candidate_results, threshold=DEFAULT_AUDIO_DIFF_THRESHOLD):
    baseline_by_index = _results_by_index(baseline_results)
    candidate_by_index = _results_by_index(candidate_results)
    anomalies = []
    for index in sorted(set(baseline_by_index) & set(candidate_by_index)):
        baseline_audio = safe_float(baseline_by_index[index]["metrics"].get("audio_seconds"))
        candidate_audio = safe_float(candidate_by_index[index]["metrics"].get("audio_seconds"))
        if baseline_audio <= 0:
            continue
        change_ratio = abs(candidate_audio - baseline_audio) / baseline_audio
        if change_ratio > threshold:
            anomalies.append(
                {
                    "index": index,
                    "baseline_audio_seconds": baseline_audio,
                    "candidate_audio_seconds": candidate_audio,
                    "change_percent": change_ratio * 100,
                }
            )
    return anomalies


def pick_benchmark_blocks(script_file, max_chars, limit):
    _paragraphs, blocks = read_paragraph_blocks(script_file, max_chars)
    return blocks[:limit]


def build_payload(text):
    return build_voxcpm_payload(
        text=text,
        model_id=MODEL_ID,
        prompt_text=PROMPT_TEXT,
        reference_wav=REFERENCE_WAV,
        cfg_value=CFG_VALUE,
        inference_timesteps=INFERENCE_TIMESTEPS,
        normalize=NORMALIZE,
    )


def run_benchmark(blocks, server_url, warmup_count=0):
    results = []
    for index, text in enumerate(blocks, start=1):
        is_warmup = index <= warmup_count
        label = "Warmup" if is_warmup else "Midiendo"
        print(f"\n[{index}/{len(blocks)}] {label} ({len(text)} caracteres)...")
        started = time.perf_counter()
        try:
            _wav_bytes, metrics = generate_wav_bytes_with_metrics(build_payload(text), server_url)
            local_seconds = time.perf_counter() - started
            ratio = metric_ratio(metrics)
            results.append(
                {
                    "ok": True,
                    "index": index,
                    "text_chars": len(text),
                    "local_seconds": local_seconds,
                    "metrics": metrics,
                    "ratio": ratio,
                    "warmup": is_warmup,
                }
            )
            print(
                f"  inferencia {safe_float(metrics.get('inference_seconds')):.2f}s | "
                f"audio {safe_float(metrics.get('audio_seconds')):.2f}s | "
                f"ratio {ratio:.2f}x | request {safe_float(metrics.get('request_seconds')):.2f}s"
            )
        except Exception as exc:
            results.append(
                {"ok": False, "index": index, "text_chars": len(text), "error": str(exc), "warmup": is_warmup}
            )
            print(f"  ERROR: {exc}")
    return results


def print_summary(summary):
    print("\nResumen benchmark:")
    print(f"  bloques OK: {summary['completed']} | warmup: {summary['warmups']} | fallidos: {summary['failed']}")
    print(f"  inferencia promedio: {summary['avg_inference_seconds']:.2f}s")
    print(f"  audio promedio: {summary['avg_audio_seconds']:.2f}s")
    print(f"  ratio promedio inferencia/audio: {summary['avg_ratio']:.2f}x")
    print(f"  ratio mediano inferencia/audio: {summary['median_ratio']:.2f}x")
    print(f"  cache total medido: {summary['total_cache_seconds']:.2f}s")
    print(f"  inferencia total: {summary['total_inference_seconds']:.2f}s")
    print(f"  request servidor total: {summary['total_request_seconds']:.2f}s")


def print_comparison(comparison, anomalies=None):
    print("\nComparacion:")
    print(
        f"  ratio mediano inferencia/audio: {comparison['baseline_median_ratio']:.2f}x -> "
        f"{comparison['candidate_median_ratio']:.2f}x "
        f"({comparison['median_ratio_improvement_percent']:.1f}% mejora)"
    )
    print(
        f"  ratio promedio inferencia/audio: {comparison['baseline_avg_ratio']:.2f}x -> "
        f"{comparison['candidate_avg_ratio']:.2f}x"
    )
    print(
        f"  inferencia promedio: {comparison['baseline_avg_inference_seconds']:.2f}s -> "
        f"{comparison['candidate_avg_inference_seconds']:.2f}s "
        f"({comparison['inference_improvement_percent']:.1f}% mejora)"
    )
    if anomalies:
        print("  advertencias audio_seconds:")
        for anomaly in anomalies:
            print(
                f"    bloque {anomaly['index']}: {anomaly['baseline_audio_seconds']:.2f}s -> "
                f"{anomaly['candidate_audio_seconds']:.2f}s "
                f"({anomaly['change_percent']:.1f}% cambio)"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark de inferencia VoxCPM sin guardar audios.")
    parser.add_argument("--script-file", default=SCRIPT_FILE)
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--limit", type=int, default=DEFAULT_BLOCK_LIMIT)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--audio-diff-threshold", type=float, default=DEFAULT_AUDIO_DIFF_THRESHOLD)
    parser.add_argument(
        "--compare-json",
        nargs=2,
        metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
        help="Compara dos salidas JSON, por ejemplo normal vs optimize.",
    )
    args = parser.parse_args()

    if args.compare_json:
        baseline = json.loads(Path(args.compare_json[0]).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.compare_json[1]).read_text(encoding="utf-8"))
        anomalies = find_audio_anomalies(
            baseline.get("results", []),
            candidate.get("results", []),
            threshold=args.audio_diff_threshold,
        )
        print_comparison(compare_summaries(baseline["summary"], candidate["summary"]), anomalies=anomalies)
        return

    try:
        health = check_server(args.server_url, timeout=args.health_timeout)
    except VoxCPMServerError as exc:
        print(f"ERROR: {exc}")
        print("\nAbre otra terminal y espera a que diga 'Servidor listo':")
        print("  python voxcpm_server.py")
        print("\nPara probar optimize despues:")
        print("  python voxcpm_server.py --optimize")
        raise SystemExit(1) from exc
    print(f"Servidor VoxCPM listo: {health.get('model_id', MODEL_ID)}")
    print(f"Optimize: {health.get('optimize')}")
    print(f"Caches de voz activas: {health.get('prompt_caches', 0)}")

    block_count = args.limit + max(args.warmup_count, 0)
    blocks = pick_benchmark_blocks(args.script_file, MAX_CHARS, block_count)
    print(f"Benchmark: {len(blocks)} bloques desde {args.script_file} ({args.warmup_count} warmup)")
    results = run_benchmark(blocks, args.server_url, warmup_count=max(args.warmup_count, 0))
    summary = summarize_metrics(results)
    print_summary(summary)

    if args.output_json:
        output = {"server": health, "summary": summary, "results": results}
        Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON guardado: {args.output_json}")


if __name__ == "__main__":
    main()
