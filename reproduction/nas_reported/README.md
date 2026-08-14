# NB301 Analysis Reproduction

This folder contains the clean analysis code used with the NB301 score and
refinement pipeline in `reproduction/nas/`.

After completing the main pipeline, run the complete reported-analysis chain on
GPU0. The bundled NB301 runtime is used by default:

```bash
OP_SCORE_ROOT=/path/to/nb301_main/operation_scores \
MAIN_OUT_ROOT=/path/to/nb301_main \
OUT_ROOT=/path/to/nb301_reported \
bash reproduction/nas_reported/run_nb301_reported_analysis.sh 0
```

This runs component and geometry analyses, axis export, paired controls, subset
stability, balanced-pool readout, the combined summary, and all five NAS figure
panels. Set `RUN_SUBSET_STABILITY=0` only when intentionally skipping the
longer subset batch. `NAS_RUNTIME_ROOT` remains an optional override for an
equivalent external runtime.

## Component Ablation

After computing the operation scores, run the component ablation on GPU0:

```bash
OP_SCORE_ROOT=/path/to/nb301_main/operation_scores \
FULL_SELECTION_DETAILS=/path/to/nb301_main/caches/full_proxy_pool_details.json \
OUT_ROOT=/path/to/nb301_reported_component_ablation \
bash reproduction/nas_reported/run_nb301_component_ablation.sh 0
```

The launcher evaluates every candidate as a single proxy, then reads the full
row's label-free selection record and uses exactly that retained proxy set for
both the direct raw mean and raw-score task-conditioned refinement. This keeps
proxy selection fixed while isolating disentanglement. The factorized full row
comes from the main `nas` run.

## Proxy Geometry

Recompute correlation and effective rank directly from the operation scores:

```bash
python reproduction/nas_reported/analyze_nb301_proxy_geometry.py \
  --op-score-root /path/to/nb301_main/operation_scores \
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
OP_SCORE_ROOT=/path/to/nb301_main/operation_scores \
OUT_ROOT=/path/to/nb301_budgeted_stability \
bash reproduction/nas/scripts/run_nb301_budgeted_stability.sh 0
```

For the current paper configuration, use the same rounded refinement schedule
for every controlled subset:

```bash
OP_SCORE_ROOT=/path/to/nb301_main/operation_scores \
OUT_ROOT=/path/to/nb301_subset_current_method \
SUBSET_AXIS_STEPS=30 SUBSET_TOTAL_STEPS=60 SUBSET_RESIDUAL_SCALE=1.00 \
AXIS_CALIBRATION_LR=0.015 COMPONENT_CORRECTION_LR=0.060 \
COMPONENT_CORRECTION_SCALE=1.00 \
OPERATION_TOPK_COUNT=5 SEED=9000 \
bash reproduction/nas/scripts/run_nb301_budgeted_stability.sh 0
```

For a two-GPU run, start the same command twice with
`SUBSET_SHARD_INDEX=0 SUBSET_SHARD_COUNT=2` on GPU0 and
`SUBSET_SHARD_INDEX=1 SUBSET_SHARD_COUNT=2` on GPU1. Both shards write
disjoint subset directories under the shared output root.

The launcher first derives a manifest from the current full-pool selection
record. It samples three size-3 and three size-5 subsets from the complete
candidate pool and independently samples the same counts from the currently
retained pool, using seed `20260615`. Refinement and control launchers consume
that same manifest, so subset identities cannot drift between methods.

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

The paired main controls read both proxy sets from the cache-builder selection
details produced by the same run. This prevents a fresh gate result and its
paired aggregation controls from silently using different proxy identities.

## Main-Table Rows

Rebuild the four current NAS main-table rows directly from the bundled frozen
balanced-pool and runtime evidence:

```bash
python reproduction/nas_reported/build_nb301_main_table_rows.py \
  --balanced-pool-csv reproduction/nas_reported/evidence/nb301_current_balanced_pool_results.csv \
  --runtime-csv reproduction/nas_reported/evidence/nb301_current_runtime.csv \
  --budget-control-result reproduction/nas_reported/evidence/artifacts/controls/budgeted_log_rank.json \
  --full-control-result reproduction/nas_reported/evidence/artifacts/controls/full_log_rank.json \
  --budget-selection-details reproduction/nas_reported/evidence/artifacts/main/budgeted_proxy_subset_details.json \
  --full-selection-details reproduction/nas_reported/evidence/artifacts/main/full_proxy_pool_details.json \
  --output-json /path/to/nb301_main_table_rows.json
```

ProxyDiff score and refinement costs are reported in GPU-hours. The paired
control searches reuse computed operation scores and execute on CPU, so their
post-score cost is reported separately as measured CPU-hours rather than as a
nominal GPU charge.

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
Canonical-LF SHA-256 f0c08e774e1c714da6b222ce203c7eff2f8ded8e77ff92bd982f015ce3b81763

evidence/balanced_pools/balanced3x1000_splits.json
SHA-256 2af3b354dd9ae481d7f6703a82a3062a9c6ba95b59eaa8911ed328fe18dd0624
```

The reference-pool wrapper preserves the exact frozen paper evaluation pool,
while its ordered `records` payload is unchanged from the original baseline
artifact. The
canonical JSON hash of that payload is
`76b51d61d0f100fe3345efdf6cbaed85d878b7774861e4cdefb599bd7de6a175`.
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
  --main-summary-csv /path/to/nb301_summary.csv \
  --component-direct-json /path/to/direct_free_decode.json \
  --raw-refinement-json /path/to/raw_refinement_free_decode.json \
  --geometry-csv /path/to/proxy_geometry.csv \
  --subset-summary-csv /path/to/nb301_budgeted_stability_summary.csv \
  --runtime-csv /path/to/nb301_runtime.csv \
  --out-csv /path/to/nb301_reported_results_summary.csv
```

The summarizer is source-driven and rejects missing evidence instead of
substituting bundled or hard-coded results.

The current clean main-table and fixed-subset summaries are bundled as:

```text
evidence/nb301_current_main_results.csv
evidence/nb301_current_main_summary.csv
evidence/nb301_current_runtime.csv
evidence/nb301_current_subset_summary.csv
evidence/nb301_current_subset_results.csv
evidence/nb301_current_subset_manifest.json
evidence/artifact_sha256.txt
```

The `evidence/artifacts/` tree contains the validated 19-score provenance
manifest, source decode, selection, control, geometry, component-ablation, and
all 36 subset-result artifacts consumed by those tables. Verify their hashes
with the top-level package checker before using the bundled numbers.

## Figures

Install Pillow in the analysis environment, then render the five NAS panels
directly from a completed run:

```bash
python reproduction/nas_reported/plot_nb301_reported_results.py \
  --reported-summary /path/to/nb301_reported_results_summary.csv \
  --main-summary /path/to/nb301_main/nb301_summary.csv \
  --full-decode /path/to/nb301_main/full_proxy_pool_free_decode.json \
  --axis-weights /path/to/nb301_reported/axis_calibration_weights.csv \
  --axis-profiles /path/to/nb301_reported/axis_proxy_profiles.csv \
  --subset-results /path/to/nb301_subset_comparison.csv \
  --output-dir /path/to/nb301_figures
```

The trajectory panel requires exact saved checkpoints at steps
`0/10/20/30/40/50/60`; it fails on incomplete evidence instead of interpolating
points. The script produces the NAS component-ablation, geometry, refinement,
axis-calibration, and subset-stability panels without loading pruning code or
private manuscript files.
