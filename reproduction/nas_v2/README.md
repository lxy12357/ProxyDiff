# NB301 ProxyDiff v2 Reproduction

This folder reproduces the NB301 ProxyDiff main-table rows from operation-score
computation through final free-decoded architecture evaluation.

Fresh-score outputs verified on the hdd GPU0 environment:

| row | score prior | axis-calibrated selection | component-corrected refinement | rank path |
|---|---:|---:|---:|---|
| full proxy pool | 93.723389 | 94.349678 | 94.512589 | 180 -> 5 -> 1 |
| 3-proxy subset | 94.250977 | 94.425629 | 94.425629 | 12 -> 2 -> 2 |

Both rows are non-decreasing by NB301 surrogate accuracy, and the final full
proxy-pool result is higher than the final 3-proxy result.

## Pipeline

1. Compute Zero-Cost-PT operation-ablation scores for the NB301 proxy pool.
2. Build ProxyDiff score caches for the full proxy pool and the 3-proxy subset.
3. Run task-conditioned refinement.
4. Free-decode and evaluate the selected NB301 architectures.

Run the full pipeline on hdd GPU0:

```bash
bash reproduction/nas_v2/scripts/run_nb301_v2_pipeline.sh 0
```

The default hdd paths are:

```bash
REPRO=/hdd/xiaoyun/ProxyDARTS/Reproduction
NAS_RUNTIME_ROOT=/hdd/xiaoyun/ProxyDARTS/Reproduction/nas_runtime/ZeroCostNAS
SCORE_PY=/hdd/xiaoyun/conda_envs/proxydarts-repro/bin/python
REFINEMENT_PY=/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main
```

By default the launcher writes fresh operation scores under
`${OUT_ROOT}/operation_scores`.  Set `OP_SCORE_ROOT` only when intentionally
using a precomputed operation-score directory.

Override them only when rebuilding the environment:

```bash
REPRO=/path/to/reproduction/root \
NAS_RUNTIME_ROOT=/path/to/nas_runtime/ZeroCostNAS \
SCORE_PY=/path/to/score/python \
REFINEMENT_PY=/path/to/refinement/python \
OUT_ROOT=/path/to/output \
bash reproduction/nas_v2/scripts/run_nb301_v2_pipeline.sh 0
```

To compute only operation scores:

```bash
STOP_AFTER_OPERATION_SCORES=1 bash reproduction/nas_v2/scripts/run_nb301_v2_pipeline.sh 0
```

To rerun refinement from an existing cache:

```bash
bash reproduction/nas_v2/scripts/run_nb301_v2_refine_from_cache.sh \
  full_proxy_pool /path/to/full_proxy_pool_proxydiff_cache.pt 0

bash reproduction/nas_v2/scripts/run_nb301_v2_refine_from_cache.sh \
  three_proxy_subset /path/to/three_proxy_subset_proxydiff_cache.pt 0
```

Run the fixed-subset stability batch after operation scores are available:

```bash
OP_SCORE_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores \
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_subset_stability \
bash reproduction/nas_v2/scripts/run_nb301_v2_subset_stability.sh 0
```

The fixed-subset launcher uses the reduced-proxy v2 setting by default
(`REDUCED_AXIS_STEPS=5`, `REDUCED_TOTAL_STEPS=40`,
`REDUCED_RESIDUAL_SCALE=1.0`).  The retained-pool 5-proxy rows use
`RETAINED_K5_AXIS_STEPS=35`, `RETAINED_K5_TOTAL_STEPS=95`, and
`RETAINED_K5_RESIDUAL_SCALE=0.75`; these variables can be overridden for
controlled reruns.

The refine-from-cache launcher defaults to the verified v2 settings:

| row | axis calibration steps | total training steps | residual axis scale | component correction lr |
|---|---:|---:|---:|---:|
| full proxy pool | 50 | 115 | 1.00 | 0.10 |
| 3-proxy subset | 10 | 45 | 0.50 | 0.025 |

The defaults use fixed rounded parameter values from a small reproduction grid
and seed `9000`.

Summarize a completed run:

```bash
python reproduction/nas_v2/src/summarize_nb301_v2_results.py \
  --out_root /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main \
  --out_csv /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/nb301_v2_summary.csv
```

Summarize the reproduced NB301 analysis values:

```bash
python reproduction/nas_v2/src/summarize_nb301_v2_reported_results.py \
  --v2_summary_csv /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/nb301_v2_summary.csv \
  --subset_summary_csv /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_subset_stability/nb301_v2_subset_stability_summary.csv \
  --out_csv /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/nb301_v2_reported_summary.csv
```

## Code Layout

- `scripts/run_nb301_v2_pipeline.sh`: end-to-end score, cache, refinement, and
  summary launcher.
- `scripts/run_nb301_v2_refine_from_cache.sh`: refinement-only launcher from a
  precomputed ProxyDiff cache.
- `scripts/run_nb301_v2_subset_stability.sh`: fixed-subset stability launcher.
- `src/compute_nb301_zcpt_operation_scores.py`: NB301 ZCPT operation-ablation
  score computation.
- `src/proxydiff_nas.py`: proxy score alignment, factorization, and cache
  writer.
- `src/run_nb301_proxy_refinement.py`: clean wrapper around the NB301
  task-conditioned refinement runtime.
- `src/_proxydiff_nb301_refinement_impl.py`: bundled NB301 refinement evaluator.
- `src/reevaluate_free_selected_arch.py`: legal NB301 free-decode and surrogate
  evaluation.
- `src/summarize_nb301_v2_results.py`: deterministic parser for v2 output
  directories.
- `src/summarize_nb301_v2_reported_results.py`: compact summary of NB301
  analysis values reproduced by the v2 package.
- `src/summarize_nb301_v2_subset_results.py`: fixed-subset stability parser.
- `configs/`: refinement config template.

The external hdd runtime supplies NB301, DARTS search-space code, Zero-Cost-PT
dependencies, surrogate model assets, and the fixed architecture pool.
