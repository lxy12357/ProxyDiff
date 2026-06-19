# NAS-Bench-301 Reproduction

This folder reproduces the NB301 ProxyDiff rows in the main table.

Targets:

- ProxyDiff full-proxy-pool row: average rank 1.0, surrogate accuracy 94.6956.
- ProxyDiff three-proxy-subset row: average rank 1.3, surrogate accuracy 94.5154.

The pipeline has three stages:

1. Compute Zero-Cost-PT operation-ablation scores for the non-degenerate proxies.
2. Build ProxyDiff score caches for the full proxy pool after utility gating and for the fixed three-proxy subset.
3. Run task-conditioned refinement and free-decode the selected NB301 architecture.

The ProxyDiff-specific code needed for these stages is organized as:

- `scripts/`: runnable entry points.
- `src/fixedpool_nb301_zcpt_single_proxy.py`: ZCPT operation-ablation scorer.
- `src/proxydiff_nas.py`: proxy rank-alignment, factorization, and cache writer.
- `src/run_nb301_proxy_refinement.py`: task-conditioned NB301 refinement.
- `src/_proxydiff_nb301_refinement_impl.py`: bundled ProxyDiff NB301 refinement
  evaluator used by the public runner.
- `src/reevaluate_free_selected_arch.py`: legal NB301 free-decode and surrogate evaluation.
- `src/collect_refinement_summaries.py`: compact run summary collector.
- `src/postprocess_nb301_clean_run.py`: clean artifact aliasing and run report
  generator for a completed pipeline output root.
- `src/summarize_nb301_main_results.py`: deterministic parser for the archived
  paper-row NB301 ProxyDiff evidence.
- `configs/`: NAS refinement configuration templates.

The launcher locates `src/` and `configs/` relative to its own path, so the same
folder works when checked out as `reproduction/nas/` in the public repository or
as `github/nas/` in the manuscript workspace.
The only remaining external requirements are data/model assets and third-party
baseline packages listed below.

The folder still expects the hdd experiment environment to provide the external
NAS runtime dependencies: the DARTS/NB301 search stack, NB301 surrogate API,
Zero-Cost-PT, and the third-party proxy implementations used by the scorer.
Those are external baselines and runtime assets, not ProxyDiff code.

Run on hdd GPU0:

```bash
bash reproduction/nas/scripts/run_nb301_pipeline.sh 0
```

The launcher defaults to the hdd runtimes that reproduce the recorded NB301
score and refinement stages.  Override them only when rebuilding the environment:

```bash
SCORE_PY=/path/to/score/python \
REFINEMENT_PY=/path/to/refinement/python \
bash reproduction/nas/scripts/run_nb301_pipeline.sh 0
```

The cache writer uses `--cache-device auto` by default, which writes CUDA
tensors when CUDA is available and falls back to CPU otherwise.

Rerun only task-conditioned refinement from an existing ProxyDiff cache:

```bash
bash reproduction/nas/scripts/run_nb301_refine_from_cache.sh full_proxy_pool /path/to/cache.pt 0
```

Set `STRICT_HISTORICAL_ENV=1` to use only the core environment knobs recorded
for the NB301 run plus the paper learning rates.  This is an isolation path for
rerunning refinement when the score/cache has already been verified.

Summarize paper-row evidence:

```bash
python reproduction/nas/src/summarize_nb301_main_results.py \
  --repo_root . \
  --out_csv Manuscript/IoTJ/results/clean_repro_nas_main_evidence/nb301_main_summary_clean.csv
```

Post-process a completed fresh pipeline run:

```bash
python reproduction/nas/src/postprocess_nb301_clean_run.py \
  --out_root /hdd/xiaoyun/ProxyDiff_Repro/nb301_main \
  --nas_runtime_root /path/to/nas_runtime
```

Important paths can be overridden:

```bash
REPRO=/hdd/xiaoyun/ProxyDARTS/Reproduction \
NAS_RUNTIME_ROOT=/path/to/nas_runtime \
CLEAN_REPO_ROOT=/path/to/reproduction_package \
SCORE_PY=/path/to/score/python \
REFINEMENT_PY=/path/to/refinement/python \
OUT_ROOT=/hdd/xiaoyun/ProxyDiff_Repro/nb301_main \
bash reproduction/nas/scripts/run_nb301_pipeline.sh 0
```
