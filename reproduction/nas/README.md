# NB301 ProxyDiff Reproduction

This folder reproduces the NB301 ProxyDiff main-table rows from operation-score
computation through final free-decoded architecture evaluation.

Fresh-score outputs verified on a CUDA GPU0 environment:

| row | score prior | axis-calibrated selection | component-corrected refinement | rank path |
|---|---:|---:|---:|---|
| full proxy pool | 94.196304 | 94.621422 | 94.759773 | 16 -> 1 -> 1 |
| budgeted 3-proxy subset | 94.079178 | 94.555138 | 94.638809 | 35 -> 1 -> 1 |

Both rows improve strictly at axis calibration and component correction, and
the final full-pool result is higher than the final budgeted result.
Both rows use one label-free gate. In the unconstrained full pool, it retains
the consensus backbone (`l2_norm`, `nwot`, `zen`, `zico`, `near`, `jacob`,
`swap`, `meco`) and automatically admits `synflow`. With budget 3, the same
budget-constrained consensus objective selects `jacob`, `snip`, and `zico`.

Measured end-to-end row costs are `6.54 + 0.06` GPU-hours for the full pool
and `1.39 + 0.05` GPU-hours for the budgeted row. The first term includes all
candidate scores required by the corresponding label-free gate (19 for the
full pool and the predeclared 14-member short-score pool for the budgeted
gate); the second term is cache construction, refinement, and free decoding.

## Pipeline

1. Compute Zero-Cost-PT operation-ablation scores for the NB301 proxy pool.
2. Build ProxyDiff score caches for the full proxy pool and budgeted subset.
   Exact-tie rank utilities and spectral effective rank determine the
   consensus-core size. A declared proxy budget constrains this core only when
   needed. Candidate complements are evaluated in the fixed-core residual
   representation using conditional effective-rank gain, operation-selection
   change, effective context support, and normal/reduction consistency. The
   positive group above the largest observed evidence gap is admitted, and the
   same residual representation is passed to refinement.
3. Run task-conditioned refinement.
4. Free-decode and evaluate the selected NB301 architectures.

## Setup

Clone the pinned public repositories used by the operation scorers:

```bash
bash reproduction/nas/scripts/setup_nb301_score_dependencies.sh
```

The script creates the ignored directory `reproduction/nas/dependencies/`.
Set `NAS_DEPENDENCY_ROOT` only when those repositories are stored elsewhere.

The minimal NB301 runtime used by refinement and surrogate evaluation is
bundled under `runtime/ZeroCostNAS`. The launcher verifies its source manifest
and benchmark payload before every run. `NAS_RUNTIME_ROOT` is only needed to
override this default intentionally.

The exact fixed architecture pool is bundled at
`assets/arch_dataset_20cell_c36.pt` (SHA-256
`852c3187ef2645469b52cfb94e9f6fa9e7b1859814f17727f181efe390c16f37`).
NASBench-301 downloads its public surrogate models on first use under a file
lock, so parallel refinement jobs do not race on the same archive. To reuse an
existing model installation, set `NB301_MODELS_ROOT` to the directory that
directly contains `xgb_v1.0/` and `lgb_runtime_v1.0/`.

CIFAR-10 is likewise downloaded under a file lock on first use. Set
`CIFAR10_DATA_ROOT` to a directory that contains `cifar-10-batches-py/` to
reuse an existing copy. Both paths are data dependencies, not method settings.

Create the standard operation-score environment:

```bash
conda env create -n proxydiff-nas-standard-scores \
  -f reproduction/nas/environment-standard-scores.yml
```

ZiCo uses a separate pinned CUDA runtime:

```bash
conda env create -n proxydiff-nas-zico \
  -f reproduction/nas/environment-zico-score.yml
```

Create the pinned refinement environment separately:

```bash
conda env create -n proxydiff-nas-refinement \
  -f reproduction/nas/environment-refinement.yml
```

Set the Python executables for the standard score, ZiCo score, and refinement
environments. The verified refinement trajectory uses Python `3.7.16`, PyTorch
`1.8.0+cu111`, and torchvision `0.9.0+cu111`; changing this runtime can alter
the learned refinement trajectory even when the score cache and seed are
identical. The launcher automatically uses the refinement environment's
`lib/` directory when it contains the required CUDA libraries.
`REFINEMENT_LD_LIBRARY_PATH` remains available as an explicit override.

## End-to-End Run

Run the full pipeline on GPU0:

```bash
SCORE_PY=/path/to/proxydiff-nas-standard-scores/bin/python \
ZICO_SCORE_PY=/path/to/proxydiff-nas-zico/bin/python \
REFINEMENT_PY=/path/to/refinement/python \
OUT_ROOT=/path/to/nb301_main \
bash reproduction/nas/scripts/run_nb301_pipeline.sh 0
```

The launcher writes fresh operation scores under
`${OUT_ROOT}/operation_scores`.  Set `OP_SCORE_ROOT` only when intentionally
using a precomputed operation-score directory. A complete fresh run also
writes `nb301_runtime.csv`, `nb301_summary.csv`, and their JSON/log evidence
under `${OUT_ROOT}`.

Before cache construction, the launcher validates all 19 score artifacts,
including metric identity, seed, batch size, operation-ablation mode, score
shape, and the scorer revision family used by that metric. It also writes one
combined `operation_score_manifest.json`, so a partially mixed score directory
is rejected rather than silently reused.

To compute only operation scores:

```bash
SCORE_PY=/path/to/proxydiff-nas-standard-scores/bin/python \
ZICO_SCORE_PY=/path/to/proxydiff-nas-zico/bin/python \
OUT_ROOT=/path/to/nb301_main \
STOP_AFTER_OPERATION_SCORES=1 \
bash reproduction/nas/scripts/run_nb301_pipeline.sh 0
```

To rerun refinement from an existing cache:

```bash
bash reproduction/nas/scripts/run_nb301_refinement.sh \
  full_proxy_pool /path/to/full_proxy_pool_proxydiff_cache.pt 0

bash reproduction/nas/scripts/run_nb301_refinement.sh \
  budgeted_proxy_subset /path/to/budgeted_proxy_subset_proxydiff_cache.pt 0
```

Run the fixed-subset stability batch after operation scores are available:

```bash
OP_SCORE_ROOT=/path/to/nb301_main/operation_scores \
OUT_ROOT=/path/to/nb301_budgeted_stability \
bash reproduction/nas/scripts/run_nb301_budgeted_stability.sh 0
```

The fixed-subset launcher uses one controlled schedule for all subsets:
`SUBSET_AXIS_STEPS=30`, `SUBSET_TOTAL_STEPS=60`, and
`SUBSET_RESIDUAL_SCALE=1.00`. To split the 12 fixed subsets across two GPUs,
launch shard `0/2` on GPU0 and shard `1/2` on GPU1 with the same `OUT_ROOT`,
then run `src/summarize_nb301_budgeted_results.py` once after both finish.

The refinement launcher defaults to the verified settings:

| row | axis calibration steps | total training steps | residual axis scale | component correction lr | component correction scale |
|---|---:|---:|---:|---:|---:|
| full proxy pool | 30 | 60 | 1.00 | 0.06 | 1.00 |
| budgeted proxy subset | 30 | 60 | 1.00 | 0.06 | 1.00 |

Axis calibration uses the temporal mean of unit task-gradient directions.
Component correction uses the unnormalized stage mean of unit task-gradient
directions, so its magnitude is the measured cross-batch directional
agreement rather than a proxy-count-specific scale. Both stages use the same
settings for the full and budgeted rows. The seed and refinement sampler are
fixed to `9000`.

The launcher applies the pinned CUDA and cuDNN settings required by ZiCo and
refinement automatically.

Every cache is accompanied by a details JSON containing its SHA-256 digest,
proxy-set identity, retained proxy names, source score hashes, and factorization
metadata. The refinement-only launcher verifies this sidecar before loading the
cache. The final summarizer exits nonzero unless both rows finish at rank 1,
each stage improves, and the full-pool final accuracy exceeds the budgeted row.

The formal source artifacts consumed by the reported tables are bundled under
`../nas_reported/evidence/artifacts/`, including all 19 operation-score JSON
files and their hash-checked manifest.

Summarize a completed run:

```bash
python reproduction/nas/src/summarize_nb301_results.py \
  --out_root /path/to/nb301_main \
  --out_csv /path/to/nb301_main/nb301_summary.csv
```

Summarize the reproduced NB301 main and subset values:

```bash
python reproduction/nas/src/summarize_nb301_reported_results.py \
  --main_summary_csv /path/to/nb301_main/nb301_summary.csv \
  --runtime_csv /path/to/nb301_main/nb301_runtime.csv \
  --subset_summary_csv /path/to/nb301_budgeted_stability/nb301_budgeted_stability_summary.csv \
  --out_csv /path/to/nb301_main/nb301_reported_summary.csv
```

## Code Layout

- `scripts/run_nb301_pipeline.sh`: end-to-end score, cache, refinement, and
  summary launcher.
- `scripts/setup_nb301_score_dependencies.sh`: pinned public scorer
  dependencies.
- `scripts/run_nb301_refinement.sh`: refinement-only launcher from a
  precomputed ProxyDiff cache.
- `scripts/run_nb301_budgeted_stability.sh`: budgeted-subset stability launcher.
- `src/compute_nb301_zcpt_operation_scores.py`: NB301 ZCPT operation-ablation
  score computation. `src/run_nb301_zcpt_operation_scores.py` is the stable
  launcher used by the shell pipeline.
- `src/verify_standard_score_runtime.py`: standard score runtime check.
- `src/verify_zico_runtime.py`: pinned ZiCo CUDA runtime check.
- `src/verify_refinement_runtime.py`: exact NB301 refinement runtime check.
- `src/proxydiff_nas.py`: proxy score alignment, factorization, and cache
  writer.
- `src/run_nb301_proxy_refinement.py`: clean wrapper around the NB301
  task-conditioned refinement runtime.
- `src/_proxydiff_nb301_refinement_impl.py`: bundled NB301 refinement evaluator.
- `src/reevaluate_free_selected_arch.py`: legal NB301 free-decode and surrogate
  evaluation.
- `src/summarize_nb301_results.py`: deterministic parser for output
  directories.
- `src/summarize_nb301_reported_results.py`: compact summary of NB301 main,
  trajectory, and subset values reproduced by the package.
- `src/summarize_nb301_budgeted_results.py`: budgeted-subset stability parser.
- `tests/`: factorization, proxy-order invariance, canonical-forward, and
  proxy-selection regressions.
- `configs/`: refinement config template.
- `environment-standard-scores.yml`: pinned standard operation-score runtime.
- `environment-zico-score.yml`: pinned deterministic ZiCo runtime.
- `environment-refinement.yml`: pinned NB301 refinement runtime.
- `runtime/`: bundled, source-verified NB301 refinement runtime.
- `assets/arch_dataset_20cell_c36.pt`: fixed NB301 architecture pool used by
  every main-table and control row.
