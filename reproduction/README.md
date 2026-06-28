# ProxyDiff Reproducibility Package

This directory contains the cleaned reproduction code for the ProxyDiff paper.

The current package is organized for the NAS reproduction path:

- `nas_v2/`: NAS-Bench-301 operation scoring and ProxyDiff refinement.

The scripts are designed to run in the hdd experiment environment used for the paper. Paths can be overridden through environment variables in each launcher.

Before running experiments, check that the clean package and small evidence
files are present:

```bash
python reproduction/check_package.py
```
