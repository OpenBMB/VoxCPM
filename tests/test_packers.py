from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

PACKERS_PATH = Path(__file__).resolve().parents[1] / "src" / "voxcpm" / "training" / "packers.py"

# Load the module directly so that stubs other test modules put into ``sys.modules`` do not interfere.
spec = importlib.util.spec_from_file_location("_voxcpm_packers_under_test", PACKERS_PATH)
packers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(packers)

first_pad_position = packers.AudioFeatureProcessingPacker._first_pad_position


def _reference_first_pad_position(tokens: torch.Tensor):
    """Original ``nonzero()``-based implementation."""
    positions = (tokens == -100).nonzero(as_tuple=True)
    if positions[0].numel() == 0:
        return None
    return int(positions[0][0])


CASES = {
    "1d_no_pad": torch.tensor([5, 6, 7, 8]),
    "1d_pad_tail": torch.tensor([5, 6, 7, -100, -100]),
    "1d_pad_middle_and_tail": torch.tensor([5, -100, 7, -100]),
    "1d_pad_first": torch.tensor([-100, 1, 2]),
    "1d_empty": torch.tensor([], dtype=torch.long),
    "2d_no_pad": torch.tensor([[1, 2, 3], [4, 5, 6]]),
    "2d_pad_in_later_row": torch.tensor([[1, 2, 3], [4, 5, 6], [7, -100, 9]]),
    "2d_pad_in_several_rows": torch.tensor([[1, 2, 3], [4, -100, 6], [-100, 8, -100]]),
    "2d_pad_first_row": torch.tensor([[1, -100], [3, 4]]),
    "2d_float_pad": torch.tensor([[0.5, 1.0], [-100.0, 2.0]]),
    "3d_pad_in_later_row": torch.zeros(3, 2, 2, dtype=torch.long).index_fill(0, torch.tensor([2]), -100),
}


@pytest.mark.parametrize("name", list(CASES))
def test_first_pad_position_matches_nonzero_reference(name):
    tokens = CASES[name]

    assert first_pad_position(tokens) == _reference_first_pad_position(tokens)


def test_first_pad_position_expected_values():
    assert first_pad_position(torch.tensor([5, 6, -100, -100])) == 2
    assert first_pad_position(torch.tensor([5, 6, 7])) is None
    assert first_pad_position(torch.tensor([[1, 2], [3, -100]])) == 1
    assert first_pad_position(torch.tensor([[1, 2], [3, 4]])) is None


def test_unpad_helpers_truncate_at_first_pad():
    packer = packers.AudioFeatureProcessingPacker.__new__(packers.AudioFeatureProcessingPacker)

    tokens = torch.tensor([1, 2, 3, -100, -100])
    assert packer.unpad_text_tokens(tokens).tolist() == [1, 2, 3]
    assert packer.unpad_audio_tokens(torch.tensor([1, 2, 3])).tolist() == [1, 2, 3]
