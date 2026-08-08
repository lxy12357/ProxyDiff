# NB301 Analysis Reproduction

This folder contains the clean analysis code used with the NB301 score and
refinement pipeline in `reproduction/nas_v2/`.

After completing the main pipeline, run all core analyses on GPU0:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
OP_SCORE_ROOT=/path/to/nb301_v2_main/operation_scores \
MAIN_OUT_ROOT=/path/to/nb301_v2_main \
OUT_ROOT=/path/to/nb301_reported \
bash reproduction/nas_reported/run_nb301_reported_analysis.sh 0
```

Set `SUBSET_SUMMARY_CSV` to include a completed fixed-subset batch in the final
combined summary.

## Component Ablation

After computing the operation scores, run the component ablation on GPU0:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
OP_SCORE_ROOT=/path/to/nb301_v2_main/operation_scores \
OUT_ROOT=/path/to/nb301_reported_component_ablation \
bash reproduction/nas_reported/run_nb301_component_ablation.sh 0
```

The launcher builds and evaluates all single-proxy scores and the direct raw
multi-proxy mean. It then runs task-conditioned refinement on the same raw
proxy columns with the clean NB301 refinement runtime. The factorized full row
comes from the main `nas_v2` run.

## Proxy Geometry

Recompute correlation and effective rank directly from the operation scores:

```bash
python reproduction/nas_reported/analyze_nb301_proxy_geometry.py \
  --op-score-root /path/to/nb301_v2_main/operation_scores \
  --out-csv /path/to/nb301_reported/proxy_geometry.csv \
  --out-details-json /path/to/nb301_reported/factorization_details.json
```

## Axis Calibration

Export the oriented proxy profiles and learned axis coefficients from the main
run:

```bash
python reproduction/nas_reported/extract_nb301_axis_calibration.py \
  --factorization-details /path/to/full_proxy_pool_details.json \
  --axis-artifact /path/to/axis_calibrated_selection.pt \
  --residual-axis-scale 1.0 \
  --out-weights-csv /path/to/axis_calibration_weights.csv \
  --out-profiles-csv /path/to/axis_proxy_profiles.csv
```

## Subset Stability

The fixed 3-proxy and 5-proxy subset launcher is provided by the main package:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
OP_SCORE_ROOT=/path/to/nb301_v2_main/operation_scores \
OUT_ROOT=/path/to/nb301_v2_subset_stability \
bash reproduction/nas_v2/scripts/run_nb301_v2_subset_stability.sh 0
```

For the current paper configuration, use the same rounded refinement schedule
for every controlled subset:

```bash
NAS_RUNTIME_ROOT=/path/to/runtime/ZeroCostNAS \
OP_SCORE_ROOT=/path/to/nb301_v2_main/operation_scores \
OUT_ROOT=/path/to/nb301_subset_current_method \
SUBSET_AXIS_STEPS=30 SUBSET_TOTAL_STEPS=40 SUBSET_RESIDUAL_SCALE=0.75 \
COMPONENT_CORRECTION_LR=0.10 COMPONENT_CORRECTION_SCALE=1.00 \
OPERATION_TOPK_COUNT=5 SEED=9000 \
bash reproduction/nas_v2/scripts/run_nb301_v2_subset_stability.sh 0
```

For a two-GPU run, start the same command twice with
`SUBSET_SHARD_INDEX=0 SUBSET_SHARD_COUNT=2` on GPU0 and
`SUBSET_SHARD_INDEX=1 SUBSET_SHARD_COUNT=2` on GPU1. Both shards write
disjoint subset directories under the shared output root.

## Paired Aggregation Controls

Run AZ-style log-rank and rank-mean evolution with the same proxy sets used by
the main ProxyDiff rows:

```bash
OPERATION_SCORE_ROOT=/path/to/operation_scores \
OUTPUT_ROOT=/path/to/nb301_main_controls \
bash reproduction/nas_reported/run_nb301_main_control_search.sh
```

For the 12 fixed subset-stability sets:

```bash
OPERATION_SCORE_ROOT=/path/to/operation_scores \
OUTPUT_ROOT=/path/to/nb301_subset_controls \
bash reproduction/nas_reported/run_nb301_subset_control_search.sh
```

Both launchers use seed `9000`, population size `50`, parent count `10`, five
generations, 25 crossover children, and 25 mutation children. The search code
is CPU-only by default because operation scores are already computed.

The current paired main controls use the hard-budget selection
`fisher,jacob,synflow` for the budgeted row and the admitted nine-proxy full
set for the full row.

## Main-Table Rows

Rebuild the four current NAS main-table rows directly from the bundled frozen
balanced-pool and runtime evidence:

```bash
python reproduction/nas_reported/build_nb301_main_table_rows.py \
  --balanced-pool-csv reproduction/nas_reported/evidence/nb301_current_balanced_pool_results.csv \
  --runtime-csv reproduction/nas_reported/evidence/nb301_current_runtime.csv \
  --output-json /path/to/nb301_main_table_rows.json
```

For a newly completed score-to-result run, regenerate the runtime evidence
from the clean master logs:

```bash
python reproduction/nas_reported/summarize_nb301_runtime.py \
  --score-master-log /path/to/score_master.log \
  --full-refinement-master-log /path/to/full_refinement_master.log \
  --budget-refinement-master-log /path/to/budget_refinement_master.log \
  --output-csv /path/to/nb301_runtime.csv \
  --output-json /path/to/nb301_runtime.json
```

## Balanced-Pool Readout

The bundled reference artifacts are the exact three balanced pools used for
all NB301 main-table search rows:

```text
evidence/balanced_pools/nb301_stratified3000_pool.json
SHA-256 7818b72ffbbc50a53a9ad9b34bed7007649a438fab1205a45003238208750358

evidence/balanced_pools/balanced3x1000_splits.json
SHA-256 2af3b354dd9ae481d7f6703a82a3062a9c6ba95b59eaa8911ed328fe18dd0624
```

The reference-pool wrapper has neutral public metadata, while its ordered
`records` payload is unchanged from the original baseline artifact. The
canonical JSON hash of that payload is
`35dc489b5aa30773c98ca3d11a4807c79307083c062e5983bb2baf07cf8694d0`.
The split file is byte-identical to the original artifact. It partitions the
same 3000 records into three disjoint 1000-architecture pools using the frozen
split seed `20260614`; no pool is resampled for a new method.

Map any final search result to those fixed pools with:

```bash
python reproduction/nas_reported/evaluate_nb301_balanced_pools.py \
  --reference-pool reproduction/nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json \
  --balanced-splits reproduction/nas_reported/evidence/balanced_pools/balanced3x1000_splits.json \
  --result full_proxy_pool=/path/to/full_proxy_pool_free_decode.json \
  --result budgeted_proxy_subset=/path/to/budgeted_proxy_subset_free_decode.json \
  --output-json /path/to/balanced_pool_results.json \
  --output-csv /path/to/balanced_pool_results.csv
```

## Combined Summary

Summarize newly generated artifacts:

```bash
python reproduction/nas_reported/summarize_nb301_reported_results.py \
  --main-summary-csv /path/to/nb301_v2_summary.csv \
  --component-direct-json /path/to/direct_free_decode.json \
  --raw-refinement-json /path/to/raw_refinement_free_decode.json \
  --geometry-csv /path/to/proxy_geometry.csv \
  --subset-summary-csv /path/to/nb301_v2_subset_stability_summary.csv \
  --out-csv /path/to/nb301_reported_results_summary.csv
```

Running the summarizer without artifact arguments prints the compact bundled
evidence table.

The current clean main-table and fixed-subset summaries are bundled as:

```text
evidence/nb301_current_main_results.csv
evidence/nb301_current_runtime.csv
evidence/nb301_current_subset_summary.csv
```
