# NB301 ProxyDiff v2 Reproduction

This folder reproduces the NB301 ProxyDiff main-table rows from operation-score
computation through final free-decoded architecture evaluation.

Fresh-score outputs verified on a CUDA GPU0 environment:

| row | score prior | axis-calibrated selection | component-corrected refinement | rank path |
|---|---:|---:|---:|---|
| full proxy pool | 93.797127 | 94.597778 | 94.640305 | 144 -> 1 -> 1 |
| budgeted 4-proxy subset | 94.022614 | 94.583778 | 94.583778 | 57 -> 1 -> 1 |

The full-pool row improves strictly at both refinement stages, the budgeted row
is non-decreasing, and the final full-pool result is higher than the final
budgeted result.
Both rows use one label-free budget-constrained gate. With budget 9, it retains
the consensus backbone (`l2_norm`, `nwot`, `zen`, `zico`, `near`, `jacob`,
`swap`, `meco`) and admits `synflow`. With budget 4, it compresses the backbone
to `fisher`, `jacob`, and `synflow`, then admits `jacob_cov`.

Measured end-to-end row costs are `6.54 + 0.05` GPU-hours for the full pool
and `1.39 + 0.05` GPU-hours for the budgeted row. The first term includes all
candidate scores required by the corresponding label-free gate (19 for the
full pool and the predeclared 14-member short-score pool for the budgeted
gate); the second term is cache construction, refinement, and free decoding.

## Pipeline

1. Compute Zero-Cost-PT operation-ablation scores for the NB301 proxy pool.
2. Build ProxyDiff score caches for the full proxy pool and budgeted subset.
   The largest-gap consensus core is retained when it fits the proxy budget;
   otherwise it is compressed to a balanced score-only coreset. A single
   complement rule then combines conditional effective-rank gain, operation
   selection impact, and normal/reduction-cell coherence. The retained core's
   effective-rank efficiency supplies the bounded geometric mixing weight.
   Candidates are retained when their normalized complement score is at least
   `0.9` and they affect both cell types.
3. Run task-conditioned refinement.
4. Free-decode and evaluate the selected NB301 architectures.

## Setup

Clone the pinned public repositories used by the operation scorers:

```bash
bash reproduction/nas_v2/scripts/setup_nb301_score_dependencies.sh
```

The script creates the ignored directory `reproduction/nas_v2/dependencies/`.
Set `NAS_DEPENDENCY_ROOT` only when those repositories are stored elsewhere.

The refinement and surrogate-evaluation stages require an NB301-compatible
`ZeroCostNAS` runtime package. `NAS_RUNTIME_ROOT` is the runtime working
directory and its parent must contain the importable `ZeroCostNAS` package.
No ProxyDiff implementation is loaded from that runtime: the scorer,
factorization, gate, refinement evaluator, decoder, and summarizers are all
bundled in this folder.

The exact fixed architecture pool is bundled at
`assets/arch_dataset_20cell_c36.pt` (SHA-256
`852c3187ef2645469b52cfb94e9f6fa9e7b1859814f17727f181efe390c16f37`).
NASBench-301 downloads its public surrogate models on first use.

ZiCo operation scores use the pinned CUDA runtime in `environment-zico.yml`:

```bash
conda env create -n proxydiff-nas-zico \
  -f reproduction/nas_v2/environment-zico.yml
```

Create the pinned refinement environment separately:

```bash
conda env create -n proxydiff-nas-refinement \
  -f reproduction/nas_v2/environment-refinement.yml
```

Set the Python executables for the standard score, ZiCo score, and refinement
environments. The verified refinement trajectory uses Python `3.7.16`, PyTorch
`1.8.0+cu111`, and torchvision `0.9.0+cu111`; changing this runtime can alter
the learned refinement trajectory even when the score cache and seed are
identical. `REFINEMENT_LD_LIBRARY_PATH` is optional and is needed only when the
refinement environment does not find its CUDA libraries automatically.

## End-to-End Run

Run the full pipeline on GPU0:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
SCORE_PY=/path/to/score/python \
ZICO_SCORE_PY=/path/to/proxydiff-nas-zico/bin/python \
REFINEMENT_PY=/path/to/refinement/python \
OUT_ROOT=/path/to/nb301_v2_main \
bash reproduction/nas_v2/scripts/run_nb301_v2_pipeline.sh 0
```

The launcher writes fresh operation scores under
`${OUT_ROOT}/operation_scores`.  Set `OP_SCORE_ROOT` only when intentionally
using a precomputed operation-score directory.

To compute only operation scores:

```bash
SCORE_PY=/path/to/score/python \
ZICO_SCORE_PY=/path/to/proxydiff-nas-zico/bin/python \
OUT_ROOT=/path/to/nb301_v2_main \
STOP_AFTER_OPERATION_SCORES=1 \
bash reproduction/nas_v2/scripts/run_nb301_v2_pipeline.sh 0
```

To rerun refinement from an existing cache:

```bash
bash reproduction/nas_v2/scripts/run_nb301_v2_refine_from_cache.sh \
  full_proxy_pool /path/to/full_proxy_pool_proxydiff_cache.pt 0

bash reproduction/nas_v2/scripts/run_nb301_v2_refine_from_cache.sh \
  budgeted_proxy_subset /path/to/budgeted_proxy_subset_proxydiff_cache.pt 0
```

Run the fixed-subset stability batch after operation scores are available:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
OP_SCORE_ROOT=/path/to/nb301_v2_main/operation_scores \
OUT_ROOT=/path/to/nb301_v2_subset_stability \
bash reproduction/nas_v2/scripts/run_nb301_v2_subset_stability.sh 0
```

The fixed-subset launcher uses one controlled schedule for all subsets:
`SUBSET_AXIS_STEPS=30`, `SUBSET_TOTAL_STEPS=40`, and
`SUBSET_RESIDUAL_SCALE=0.75`. To split the 12 fixed subsets across two GPUs,
launch shard `0/2` on GPU0 and shard `1/2` on GPU1 with the same `OUT_ROOT`,
then run `src/summarize_nb301_v2_subset_results.py` once after both finish.

The refine-from-cache launcher defaults to the verified v2 settings:

| row | axis calibration steps | total training steps | residual axis scale | component correction lr | component correction scale |
|---|---:|---:|---:|---:|---:|
| full proxy pool | 30 | 40 | 1.00 | 0.10 | 1.00 |
| budgeted proxy subset | 30 | 40 | 0.75 | 0.10 | 1.00 |

The defaults use fixed rounded parameter values from a small reproduction grid
and seed `9000`. The refinement sampler is also fixed to `9000`.

The launcher applies the deterministic CUDA and cuDNN settings required by this
runtime automatically.

Summarize a completed run:

```bash
python reproduction/nas_v2/src/summarize_nb301_v2_results.py \
  --out_root /path/to/nb301_v2_main \
  --out_csv /path/to/nb301_v2_main/nb301_v2_summary.csv
```

Summarize the reproduced NB301 main and subset values:

```bash
python reproduction/nas_v2/src/summarize_nb301_v2_reported_results.py \
  --v2_summary_csv /path/to/nb301_v2_main/nb301_v2_summary.csv \
  --subset_summary_csv /path/to/nb301_v2_subset_stability/nb301_v2_subset_stability_summary.csv \
  --out_csv /path/to/nb301_v2_main/nb301_v2_reported_summary.csv
```

## Code Layout

- `scripts/run_nb301_v2_pipeline.sh`: end-to-end score, cache, refinement, and
  summary launcher.
- `scripts/setup_nb301_score_dependencies.sh`: pinned public scorer
  dependencies.
- `scripts/run_nb301_v2_refine_from_cache.sh`: refinement-only launcher from a
  precomputed ProxyDiff cache.
- `scripts/run_nb301_v2_subset_stability.sh`: fixed-subset stability launcher.
- `src/compute_nb301_zcpt_operation_scores.py`: NB301 ZCPT operation-ablation
  score computation. `src/run_nb301_zcpt_operation_scores.py` is the stable
  launcher used by the shell pipeline.
- `src/verify_zico_runtime.py`: pinned ZiCo CUDA runtime check.
- `src/verify_refinement_runtime.py`: exact NB301 refinement runtime check.
- `src/proxydiff_nas.py`: proxy score alignment, factorization, and cache
  writer.
- `src/run_nb301_proxy_refinement.py`: clean wrapper around the NB301
  task-conditioned refinement runtime.
- `src/_proxydiff_nb301_refinement_impl.py`: bundled NB301 refinement evaluator.
- `src/reevaluate_free_selected_arch.py`: legal NB301 free-decode and surrogate
  evaluation.
- `src/summarize_nb301_v2_results.py`: deterministic parser for v2 output
  directories.
- `src/summarize_nb301_v2_reported_results.py`: compact summary of NB301 main,
  trajectory, and subset values reproduced by the v2 package.
- `src/summarize_nb301_v2_subset_results.py`: fixed-subset stability parser.
- `configs/`: refinement config template.
- `environment-zico.yml`: pinned deterministic ZiCo score runtime.
- `environment-refinement.yml`: pinned NB301 refinement runtime.
- `assets/arch_dataset_20cell_c36.pt`: fixed NB301 architecture pool used by
  every main-table and control row.
