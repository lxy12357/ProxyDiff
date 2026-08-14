# Bundled NB301 Runtime

This directory contains the minimal NB301 runtime used by the ProxyDiff
refinement launcher. It is included so the public pipeline does not depend on
an external experiment checkout.

Only the NB301 search space, CIFAR data utilities, zero-cost predictor bridge,
and the fixed zero-cost benchmark file required by refinement are retained.
The source manifest in `runtime_source_sha256.txt` fixes every Python source
file. `verify_bundled_runtime.py` validates both that manifest and the benchmark
payload before a run.

The runtime is derived from the NASLib/zero-cost NAS code used by the
experiments. Apache-licensed files retain their original notices; the complete
license text is in `LICENSE-APACHE-2.0`. ProxyDiff-specific changes limit the
runtime to NB301, remove machine-specific paths, and expose semantic axis and
component parameter names.

The NAS-Bench-301 Python package downloads its official v1.0 surrogate models
on first use into `ZeroCostNAS/data/nb_models_1.0`.
