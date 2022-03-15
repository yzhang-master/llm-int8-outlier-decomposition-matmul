# llm-int8-outlier-decomposition-matmul

Technique from 2022, reimplemented in 2026. This scaffold contains the package, storage-accounting smoke test and CI harness for row-wise INT8 matmul with outlier-feature decomposition.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT
