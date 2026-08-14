# ProxyDiff Reproducibility Package

This directory contains the cleaned reproduction code for the ProxyDiff paper.

The current package is organized for the NAS reproduction path:

- `nas/`: NAS-Bench-301 operation scoring and ProxyDiff refinement.
- `nas_reported/`: NB301 component, geometry, axis-calibration, trajectory, and
  subset analysis.

The NASBench-301 source runtime is bundled. The score and refinement Conda
environments are pinned in `nas/environment-*.yml`; paths can be overridden
through environment variables in each launcher.

Before running experiments, check that the clean package and small evidence
files are present:

```bash
python reproduction/check_package.py
```
