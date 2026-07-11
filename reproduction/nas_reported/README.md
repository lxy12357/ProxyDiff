# NB301 Analysis Reproduction

This folder contains the clean analysis code used with the NB301 score and
refinement pipeline in `reproduction/nas_v2/`.

After completing the main pipeline, run all core analyses on GPU0:

```bash
OP_SCORE_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores \
MAIN_OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main \
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_reported \
bash reproduction/nas_reported/run_nb301_reported_analysis.sh 0
```

Set `SUBSET_SUMMARY_CSV` to include a completed fixed-subset batch in the final
combined summary.

## Component Ablation

After computing the operation scores, run the component ablation on GPU0:

```bash
OP_SCORE_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores \
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_reported_component_ablation \
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
  --op-score-root /hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores \
  --out-csv /hdd/xiaoyun/ProxyDiff_Repro/nb301_reported/proxy_geometry.csv \
  --out-details-json /hdd/xiaoyun/ProxyDiff_Repro/nb301_reported/factorization_details.json
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
OP_SCORE_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores \
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_subset_stability \
bash reproduction/nas_v2/scripts/run_nb301_v2_subset_stability.sh 0
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
