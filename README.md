# GEOMETRY-SCAN

The shape of the world inside a model: per-layer intrinsic dimension, anisotropy cone strength, and variance spectrum.

Hidden states at every layer sit in a space with measurable geometry. Some layers squeeze states into a thin cone around one direction; others spread them across the full width. GEOMETRY-SCAN turns that shape into numbers.

## What it does

- Intrinsic dimension via the participation ratio of covariance eigenvalues.
- Anisotropy: mean pairwise cosine between raw state vectors (cone strength).
- PC1 variance share: how much of the spread the first direction owns.
- Full report: per-layer entries, lowest-dimension layer, highest-anisotropy layer.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from geometry_scan.scan import geometry_report

# states_by_layer[i] has shape (samples, dim) for layer i
report = geometry_report(states_by_layer, names=["l0", "l1", "l2"])
for entry in report["layers"]:
    print(entry["name"], entry["intrinsic_dimension"], entry["anisotropy"])
```

## How it works

Intrinsic dimension is the participation ratio of the covariance spectrum: 1 when all variance sits on one direction, close to the ambient width when the spread is balanced. Anisotropy is computed on the raw states, so a layer whose states all point the same way scores near 1 even before centering.

## Honest scope

- Geometry is computed on the snapshots you provide; it says nothing about layers you did not record.
- Participation ratio and cone strength are related but not interchangeable measures.
- Tests use synthetic states (line, cloud, cone), so no model download is needed.

## License

MIT