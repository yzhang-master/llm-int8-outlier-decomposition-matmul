from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RowQuantized:
    values: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]


def quantize_rows(tensor: torch.Tensor) -> RowQuantized:
    """Symmetrically quantize each row along the final (feature) dimension."""
    if not tensor.is_floating_point() or tensor.ndim < 1:
        raise ValueError("expected a floating tensor with a feature dimension")
    original_shape = tuple(tensor.shape)
    rows = tensor.detach().float().reshape(-1, original_shape[-1])
    maxima = rows.abs().amax(dim=1)
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima))
    values = torch.round(rows / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return RowQuantized(values, scales, original_shape)


def dequantize_rows(tensor: RowQuantized) -> torch.Tensor:
    rows = tensor.values.float() * tensor.scales[:, None]
    return rows.reshape(tensor.original_shape)


def integer_matmul(inputs: RowQuantized, weights: RowQuantized) -> torch.Tensor:
    """Multiply INT8 row matrices with exact INT32 accumulation."""
    if inputs.values.shape[1] != weights.values.shape[1]:
        raise ValueError("input features and weight input features must match")
    return torch._int_mm(inputs.values, weights.values.transpose(0, 1))


def quantized_matmul(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute a row-quantized matmul and apply both rows' fp32 scales."""
    quantized_inputs = quantize_rows(inputs)
    quantized_weights = quantize_rows(weights)
    accumulated = integer_matmul(quantized_inputs, quantized_weights)
    output = accumulated.float() * quantized_inputs.scales[:, None]
    output = output * quantized_weights.scales[None, :]
    return output.reshape(*inputs.shape[:-1], weights.shape[0])
