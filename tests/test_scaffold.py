import pytest

from int8_outlier import quantized_weight_bytes


def test_weight_storage_includes_int8_values_and_row_scales() -> None:
    assert quantized_weight_bytes(8, 16) == 8 * 16 + 8 * 4
    with pytest.raises(ValueError, match="positive"):
        quantized_weight_bytes(0, 16)
