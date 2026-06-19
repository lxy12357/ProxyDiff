# Reproduction Package

This directory contains the public clean reproduction code.

- `nas/`: NB301 ProxyDiff score, cache, refinement, and evaluation pipeline.
- `analysis/`: reported-result checkers and figure-generation scripts.

The NAS pipeline is the executable main-result path.  The analysis scripts
operate on recorded result artifacts and are separate from the expensive
score/refinement runs.

Package check:

```bash
python reproduction/check_package.py
```
