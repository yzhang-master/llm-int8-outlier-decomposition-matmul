import pytest
import torch

from int8_outlier.quantize import (
    dequantize_rows,
    integer_matmul,
    quantize_rows,
    quantized_matmul,
)


def test_each_row_gets_an_independent_absmax_scale() -> None:
    inputs = torch.tensor([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]])
    quantized = quantize_rows(inputs)
    torch.testing.assert_close(quantized.scales, torch.tensor([1 / 127, 2 / 127, 1.0]))
    torch.testing.assert_close(
        quantized.values,
        torch.tensor([[127, -127], [127, -127], [0, 0]], dtype=torch.int8),
    )
    torch.testing.assert_close(dequantize_rows(quantized), inputs)


def test_quantization_preserves_leading_dimensions_and_clamps_to_int8() -> None:
    inputs = torch.randn(2, 3, 16)
    quantized = quantize_rows(inputs)
    assert quantized.values.shape == (6, 16)
    assert dequantize_rows(quantized).shape == inputs.shape
    assert quantized.values.min() >= -127
    assert quantized.values.max() <= 127


def test_integer_accumulation_is_exact_not_int8_wrapping() -> None:
    inputs = quantize_rows(torch.full((1, 256), 1.0))
    weights = quantize_rows(torch.full((1, 256), 1.0))
    accumulated = integer_matmul(inputs, weights)
    assert accumulated.dtype == torch.int32
    assert accumulated.item() == 127 * 127 * 256


def test_quantized_matmul_tracks_fp32_on_normal_values() -> None:
    generator = torch.Generator().manual_seed(8)
    inputs = torch.randn(3, 5, 32, generator=generator)
    weights = torch.randn(12, 32, generator=generator)
    actual = quantized_matmul(inputs, weights)
    expected = inputs @ weights.transpose(0, 1)
    assert actual.shape == expected.shape
    relative = (actual - expected).norm() / expected.norm()
    assert relative < 0.02


def test_incompatible_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="features"):
        quantized_matmul(torch.randn(2, 8), torch.randn(3, 7))
