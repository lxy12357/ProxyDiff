# Paper Analysis and Figure Reproduction

This folder is for clean scripts that regenerate reported analysis figures and
tables outside the main NAS/LLM pipelines.

## Clean Entry Points

Render all manuscript figures from the recorded analysis artifacts:

```bash
python reproduction/analysis/make_paper_figures.py
```

Check reported secondary-result numbers against the clean artifact mapping:

```bash
python reproduction/analysis/summarize_reported_results.py --repo_root .
```

This writes:

```text
Manuscript/IoTJ/results/clean_repro_analysis/secondary_reported_results_summary.csv
```

Recompute the Llama2-7B proxy-geometry analysis on the hdd server:

```bash
COMPONENT_CORR_WEIGHTS=/path/to/component_corr_para/trainable_params.pt \
bash reproduction/analysis/run_llm_score_geometry_summary.sh
```

The shell launcher defaults to the 2026-06-15 Llama2-7B 50% score tensor. Set
`COMPONENT_CORR_WEIGHTS` to the component-correction checkpoint being
inspected. Override `SIGNAL_SCORE_CACHE`, `OUT_JSON`, or `PYTHON_BIN` only when
checking a different analysis run.

If running inside the manuscript workspace rather than the public checkout,
replace `reproduction/analysis/` with `github/analysis/`.
