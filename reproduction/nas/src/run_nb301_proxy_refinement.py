"""Run NB301 ProxyDiff task-conditioned refinement from a score cache.

This runner is used after ZCPT operation-ablation scores have been converted
into a ProxyDiff refinement cache by `proxydiff_nas.py`.  The ProxyDiff
refinement evaluator is bundled in this folder; the external NAS runtime
provides the DARTS search space, data loaders, and NB301 surrogate API.
"""
import logging
import os
import shutil
import sys

import torch

NAS_RUNTIME_PACKAGE_ROOT = os.environ.get(
    "NAS_RUNTIME_PACKAGE_ROOT",
    "/hdd/xiaoyun/ProxyDARTS/Reproduction/nas_runtime",
)
sys.path.insert(0, NAS_RUNTIME_PACKAGE_ROOT)

from _proxydiff_nb301_refinement_impl import ZeroCostPredictorEvaluator
from ZeroCostNAS.predictors import ZeroCost
from ZeroCostNAS.search_spaces import get_search_space
from ZeroCostNAS.utils import get_dataset_api, get_zc_benchmark_api, setup_logger, utils


def evaluator_key(suffix):
    return f"PROXYDIFF{suffix}"


def internal_component_stage_token(stage):
    if stage == 1:
        return "axis_calib"
    if stage == 3:
        return "component_corr"
    return f"stage_{stage}"


def refinement_objective_to_runtime_mode(objective):
    objective = (objective or "").strip().lower()
    if objective in ("", "proxy_axis_component"):
        return "8"
    if objective.isdigit():
        return objective
    raise ValueError(f"Unknown REFINEMENT_OBJECTIVE={objective!r}")


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def apply_evaluator_compat_env():
    """Translate public ProxyDiff names to the bundled evaluator's internal keys."""
    direct = {
        "PROXYDIFF_REFINEMENT_CACHE": evaluator_key("_SCORE_CACHE"),
        "REFINEMENT_AXIS_WEIGHTING": evaluator_key("_METRIC_WEIGHT_STYLE"),
        "AXIS_SCALE_LAYOUT": evaluator_key("_AXIS_CALIB_LAYOUT"),
        "RESIDUAL_AXIS_SCALE": evaluator_key("_RESIDUAL_SCALE"),
        "COMPONENT_CORR_PARA_AXIS_SCALE": evaluator_key("_COMPONENT_CORR_SCALE"),
        "AXIS_CALIB_PARA_STEPS": evaluator_key("_AXIS_CALIB_STEPS"),
        "FOCUS_MASK_TOPK": evaluator_key("_PROGRESSIVE_MASK_TOPK"),
        "EVALUATE_INITIAL_SCORE": evaluator_key("_PRETRAIN_EVAL"),
        "INITIAL_SCORE_ONLY": evaluator_key("_PRETRAIN_ONLY"),
        "SAVE_STAGE_ARTIFACTS": evaluator_key("_DUMP_EPOCH_ARTIFACTS"),
        "REFINEMENT_EPOCHS": evaluator_key("_METRIC_EPOCHS"),
        "AXIS_CALIB_PARA_LR": evaluator_key("_AXIS_CALIB_LR"),
        "AXIS_CALIB_PARA_WEIGHT_DECAY": evaluator_key("_AXIS_CALIB_WEIGHT_DECAY"),
        "COMPONENT_CORR_PARA_LR": evaluator_key("_COMPONENT_CORR_LR"),
        "COMPONENT_CORR_PARA_WEIGHT_DECAY": evaluator_key("_COMPONENT_CORR_WEIGHT_DECAY"),
        "MAX_REFINEMENT_STEPS": evaluator_key("_MAX_TRAIN_STEPS"),
        "DISABLE_EDGE_NORMALIZATION": evaluator_key("_NO_EDGE_NORMALIZE"),
        "EVALUATE_DECODE_VARIANTS": evaluator_key("_EVAL_VARIANTS"),
    }
    for public_name, evaluator_name in direct.items():
        if public_name in os.environ:
            os.environ[evaluator_name] = os.environ[public_name]

    if "REFINEMENT_OBJECTIVE" in os.environ:
        os.environ[evaluator_key("_METRIC_MODE")] = refinement_objective_to_runtime_mode(
            os.environ["REFINEMENT_OBJECTIVE"]
        )

    old_axis_calib_parameter_policy = "topk_after_axis_" + "calib_" + "para"
    old_axis_calib_parameter_source = "axis_" + "calib_" + "para"
    focus_policy = os.environ.get("FOCUS_MASK_POLICY", "").strip().lower()
    if focus_policy:
        if focus_policy in {"topk_after_axis_calibration", old_axis_calib_parameter_policy}:
            os.environ[evaluator_key("_PROGRESSIVE_MASK")] = "topk_after_axis_calib"
        else:
            os.environ[evaluator_key("_PROGRESSIVE_MASK")] = focus_policy

    focus_source = os.environ.get("FOCUS_MASK_SOURCE", "").strip().lower()
    if focus_source:
        legacy_axis_component_source = "axis_calibration_with_" + "component_" + "correction"
        if focus_source in {"axis_calibrated_score", old_axis_calib_parameter_source}:
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = "axis_calib"
        elif focus_source in (
            "axis_calib_with_component_corr",
            legacy_axis_component_source,
        ):
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = "axis_calib_with_component_corr"
        else:
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = focus_source


apply_evaluator_compat_env()

score_cache = os.environ.get("PROXYDIFF_REFINEMENT_CACHE") or os.environ.get(evaluator_key("_SCORE_CACHE"))
if not score_cache or not os.path.exists(score_cache):
    raise RuntimeError("PROXYDIFF_REFINEMENT_CACHE must point to a generated ProxyDiff refinement cache.")

cache_obj = torch.load(score_cache, map_location="cpu")
metric_count = len(cache_obj)
if metric_count <= 0:
    raise RuntimeError("ProxyDiff refinement cache is empty.")

metrics = [f"zcpt_op_shared_{i}" for i in range(metric_count)]
metric_epoch_accumulate = [False for _ in metrics]
metric_mode = int(
    refinement_objective_to_runtime_mode(os.environ.get("REFINEMENT_OBJECTIVE"))
    if "REFINEMENT_OBJECTIVE" in os.environ
    else os.environ.get(evaluator_key("_METRIC_MODE"), "8")
)

config = utils.get_config_from_args()
utils.set_seed(config.seed)

if config.search_space == "nasbench301":
    fixed_arch = (
        os.environ.get("FIXED_ARCH_FILE")
        or os.environ.get(evaluator_key("_FIXED_ARCH_FILE"))
        or "/hdd/xiaoyun/ProxyDARTS/Reproduction/fixed_archs/arch_dataset_20cell_c36.pt"
    )
    save_arch_dir = os.path.join(config.save, "arch_data")
    os.makedirs(save_arch_dir, exist_ok=True)
    save_arch = os.path.join(save_arch_dir, "arch_dataset.npy")
    if not os.path.exists(save_arch):
        shutil.copyfile(fixed_arch, save_arch)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
logger.info(
    "ProxyDiff NB301 refinement: cache_metrics=%s, evaluator_metric_names=%s, cache=%s, refinement_objective=%s",
    metric_count,
    metrics,
    score_cache,
    metric_mode,
)

dataset_api = get_dataset_api(config.search_space, config.dataset)
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)
predictor = ZeroCost(method_type=metrics)
search_space = get_search_space(
    name=config.search_space,
    dataset=config.dataset,
    metric_names=metrics,
    metric_epoch_accumulate=metric_epoch_accumulate,
)
search_space.instantiate_model = False
search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

if config.search_space != "nasbench301":
    raise RuntimeError("The clean public NAS reproduction package supports only NB301.")

predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config, zc_api=zc_api)
load_all = False
load_labeled = True
test_data_file = os.path.join(config.save, "arch_data/arch_dataset.npy")

predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api, load_labeled=load_labeled)
predictor_evaluator.evaluate(
    zc_api,
    train_epoch=0,
    weight_path=None,
    step=False,
    perturbation=True,
    pruning=False,
    no_zero=True,
    arch_num=1000,
    load_all=load_all,
    metric_epoch=1,
    test_data_file=test_data_file,
    metric_mode=metric_mode,
    auto_augment=True,
)
logger.info("ZCPT operation shared-rank refinement completed.")
close_logger(logger)
