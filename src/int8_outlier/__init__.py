"""LLM.int8 row quantization and outlier-feature decomposition."""

__version__ = "0.1.0"


def quantized_weight_bytes(rows: int, columns: int) -> int:
    """Count INT8 weights plus one fp32 scale per output row."""
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    return rows * columns + rows * 4
