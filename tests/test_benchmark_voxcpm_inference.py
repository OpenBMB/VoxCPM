import benchmark_voxcpm_inference as benchmark


def test_metric_ratio_uses_inference_over_audio_seconds():
    assert benchmark.metric_ratio({"inference_seconds": 12.0, "audio_seconds": 4.0}) == 3.0


def test_metric_ratio_handles_missing_audio_seconds():
    assert benchmark.metric_ratio({"inference_seconds": 12.0}) == 0.0
    assert benchmark.metric_ratio({"inference_seconds": 12.0, "audio_seconds": 0.0}) == 0.0


def test_summarize_metrics_reports_average_and_failed_counts():
    results = [
        {
            "ok": True,
            "warmup": True,
            "metrics": {
                "inference_seconds": 100.0,
                "audio_seconds": 50.0,
                "cache_seconds": 10.0,
                "request_seconds": 110.0,
            },
        },
        {
            "ok": True,
            "metrics": {
                "inference_seconds": 10.0,
                "audio_seconds": 5.0,
                "cache_seconds": 0.5,
                "request_seconds": 10.5,
            },
        },
        {
            "ok": True,
            "metrics": {
                "inference_seconds": 6.0,
                "audio_seconds": 3.0,
                "cache_seconds": 0.0,
                "request_seconds": 6.5,
            },
        },
        {"ok": False, "error": "boom"},
    ]

    summary = benchmark.summarize_metrics(results)

    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["warmups"] == 1
    assert summary["avg_inference_seconds"] == 8.0
    assert summary["avg_audio_seconds"] == 4.0
    assert summary["avg_ratio"] == 2.0
    assert summary["median_ratio"] == 2.0
    assert summary["total_cache_seconds"] == 0.5
    assert summary["total_request_seconds"] == 17.0
    assert summary["total_inference_seconds"] == 16.0


def test_summarize_metrics_handles_no_completed_results():
    summary = benchmark.summarize_metrics([{"ok": False, "error": "boom"}])

    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["warmups"] == 0
    assert summary["avg_ratio"] == 0.0


def test_compare_summaries_reports_percent_improvement():
    comparison = benchmark.compare_summaries(
        {"median_ratio": 3.0, "avg_ratio": 3.5, "avg_inference_seconds": 12.0},
        {"median_ratio": 2.0, "avg_ratio": 2.5, "avg_inference_seconds": 9.0},
    )

    assert comparison["baseline_median_ratio"] == 3.0
    assert comparison["candidate_median_ratio"] == 2.0
    assert round(comparison["median_ratio_improvement_percent"], 1) == 33.3
    assert comparison["baseline_avg_ratio"] == 3.5
    assert comparison["candidate_avg_ratio"] == 2.5
    assert comparison["inference_improvement_percent"] == 25.0


def test_find_audio_anomalies_flags_large_audio_duration_changes():
    baseline = [
        {"ok": True, "index": 1, "metrics": {"audio_seconds": 10.0}},
        {"ok": True, "index": 2, "metrics": {"audio_seconds": 5.0}},
    ]
    candidate = [
        {"ok": True, "index": 1, "metrics": {"audio_seconds": 13.5}},
        {"ok": True, "index": 2, "metrics": {"audio_seconds": 5.5}},
    ]

    anomalies = benchmark.find_audio_anomalies(baseline, candidate, threshold=0.30)

    assert anomalies == [
        {
            "index": 1,
            "baseline_audio_seconds": 10.0,
            "candidate_audio_seconds": 13.5,
            "change_percent": 35.0,
        }
    ]


def test_find_audio_anomalies_ignores_warmup_results():
    baseline = [{"ok": True, "index": 1, "warmup": True, "metrics": {"audio_seconds": 10.0}}]
    candidate = [{"ok": True, "index": 1, "warmup": True, "metrics": {"audio_seconds": 20.0}}]

    assert benchmark.find_audio_anomalies(baseline, candidate, threshold=0.30) == []
