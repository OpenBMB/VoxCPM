"""Unit checks for pick_runtime_dtype / get_dtype consistency.

Loads src/voxcpm/model/utils.py directly to avoid the heavy voxcpm package
init. Run with: `python scripts/test_pick_runtime_dtype.py`.
"""
import importlib.util
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
UTILS = str(REPO_ROOT / "src" / "voxcpm" / "model" / "utils.py")
spec = importlib.util.spec_from_file_location("voxcpm_utils", UTILS)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

_LOW_PRECISION_DTYPES = utils._LOW_PRECISION_DTYPES
_VALID_DTYPE_OVERRIDES = utils._VALID_DTYPE_OVERRIDES
get_dtype = utils.get_dtype
pick_runtime_dtype = utils.pick_runtime_dtype


def expect(actual, expected, label):
    ok = actual == expected
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {label}: got={actual!r} expected={expected!r}")
    return ok


def expect_raises(fn, exc_type, label):
    try:
        fn()
    except exc_type as e:
        print(f"[OK ] {label}: raised {exc_type.__name__}: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: raised {type(e).__name__} not {exc_type.__name__}: {e}")
        return False
    print(f"[FAIL] {label}: no exception raised")
    return False


results = []

print("=== override set sanity ===")
results.append(expect("half" not in _VALID_DTYPE_OVERRIDES, True, "half removed from _VALID_DTYPE_OVERRIDES"))
results.append(expect("half" not in _LOW_PRECISION_DTYPES, True, "half removed from _LOW_PRECISION_DTYPES"))

print("\n=== every accepted override parses through get_dtype ===")
for dt in sorted(_VALID_DTYPE_OVERRIDES):
    try:
        torch_dtype = get_dtype(dt)
        print(f"[OK ] get_dtype({dt!r}) -> {torch_dtype}")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] get_dtype({dt!r}) raised: {e}")
        results.append(False)

print("\n=== pick_runtime_dtype: non-mps is a no-op ===")
results.append(expect(pick_runtime_dtype("cuda", "bfloat16"), "bfloat16", "cuda/bf16 untouched"))
results.append(expect(pick_runtime_dtype("cpu", "float16"), "float16", "cpu/fp16 untouched"))
results.append(expect(pick_runtime_dtype("cuda", "float32"), "float32", "cuda/fp32 untouched"))

print("\n=== pick_runtime_dtype: mps forces fp32 for low-precision ===")
os.environ.pop("VOXCPM_MPS_DTYPE", None)
results.append(expect(pick_runtime_dtype("mps", "bfloat16"), "float32", "mps/bf16 -> fp32"))
results.append(expect(pick_runtime_dtype("mps", "bf16"), "float32", "mps/bf16-alias -> fp32"))
results.append(expect(pick_runtime_dtype("mps", "float16"), "float32", "mps/fp16 -> fp32"))
results.append(expect(pick_runtime_dtype("mps", "fp16"), "float32", "mps/fp16-alias -> fp32"))
results.append(expect(pick_runtime_dtype("mps", "float32"), "float32", "mps/fp32 stays"))
results.append(expect(pick_runtime_dtype("mps", "fp32"), "fp32", "mps/fp32-alias stays"))

print("\n=== pick_runtime_dtype: VOXCPM_MPS_DTYPE override ===")
os.environ["VOXCPM_MPS_DTYPE"] = "bfloat16"
results.append(expect(pick_runtime_dtype("mps", "bfloat16"), "bfloat16", "override bf16 honored"))

os.environ["VOXCPM_MPS_DTYPE"] = "FP16"
results.append(expect(pick_runtime_dtype("mps", "bfloat16"), "fp16", "override is case-insensitive"))

os.environ["VOXCPM_MPS_DTYPE"] = "  float32  "
results.append(expect(pick_runtime_dtype("mps", "bfloat16"), "float32", "override is whitespace-trimmed"))

print("\n=== pick_runtime_dtype: 'half' is no longer a valid override ===")
os.environ["VOXCPM_MPS_DTYPE"] = "half"
results.append(
    expect_raises(
        lambda: pick_runtime_dtype("mps", "bfloat16"),
        ValueError,
        "override=half now rejected (was the bug)",
    )
)

os.environ["VOXCPM_MPS_DTYPE"] = "garbage"
results.append(
    expect_raises(
        lambda: pick_runtime_dtype("mps", "bfloat16"),
        ValueError,
        "override=garbage still rejected",
    )
)

os.environ.pop("VOXCPM_MPS_DTYPE", None)

print("\n=== summary ===")
passed = sum(results)
total = len(results)
print(f"{passed}/{total} passed")
sys.exit(0 if passed == total else 1)
