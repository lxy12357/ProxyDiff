"""Run NB301 ProxyDiff task-conditioned refinement from a score cache.

This runner is used after ZCPT operation-ablation scores have been converted
into a ProxyDiff refinement cache by `proxydiff_nas.py`.  The ProxyDiff
refinement evaluator is bundled in this folder; the external NAS runtime
provides the DARTS search space, data loaders, and NB301 surrogate API.
"""
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def install_configspace_normal_bounds_compat() -> None:
    """Allow NASBench-301 ConfigSpace JSON files to load on newer ConfigSpace."""
    try:
        from ConfigSpace.read_and_write import dictionary as cs_dictionary
    except Exception:
        return

    float_decoder = getattr(cs_dictionary, "_decode_normal_float", None)
    int_decoder = getattr(cs_dictionary, "_decode_normal_int", None)
    if float_decoder is None or getattr(float_decoder, "_proxydiff_normal_bounds_compat", False):
        return

    def _decode_normal_float_compat(item, cs, dec):
        if "lower" not in item or "upper" not in item:
            item = dict(item)
            mu = float(item.get("mu", 0.0))
            sigma = abs(float(item.get("sigma", 1.0)))
            if item.get("log", False):
                item.setdefault("lower", max(mu * 1e-3, 1e-12))
                item.setdefault("upper", max(mu * 1e3, mu + 10.0 * sigma, 1.0))
            else:
                item.setdefault("lower", mu - 10.0 * sigma)
                item.setdefault("upper", mu + 10.0 * sigma)
        return float_decoder(item, cs, dec)

    def _decode_normal_int_compat(item, cs, dec):
        if "lower" not in item or "upper" not in item:
            item = dict(item)
            mu = int(round(float(item.get("mu", 0))))
            sigma = max(1, int(round(abs(float(item.get("sigma", 1))))))
            item.setdefault("lower", mu - 10 * sigma)
            item.setdefault("upper", mu + 10 * sigma)
        return int_decoder(item, cs, dec)

    _decode_normal_float_compat._proxydiff_normal_bounds_compat = True
    _decode_normal_int_compat._proxydiff_normal_bounds_compat = True
    cs_dictionary._decode_normal_float = _decode_normal_float_compat
    if int_decoder is not None:
        cs_dictionary._decode_normal_int = _decode_normal_int_compat
    if hasattr(cs_dictionary, "HYPERPARAMETER_DECODERS"):
        cs_dictionary.HYPERPARAMETER_DECODERS["normal_float"] = _decode_normal_float_compat
        if int_decoder is not None:
            cs_dictionary.HYPERPARAMETER_DECODERS["normal_int"] = _decode_normal_int_compat


install_configspace_normal_bounds_compat()

NAS_RUNTIME_PACKAGE_ROOT = os.environ.get("NAS_RUNTIME_PACKAGE_ROOT")
if not NAS_RUNTIME_PACKAGE_ROOT:
    raise RuntimeError(
        "NAS_RUNTIME_PACKAGE_ROOT must point to the parent directory of the "
        "ZeroCostNAS runtime package."
    )
sys.path.insert(0, NAS_RUNTIME_PACKAGE_ROOT)


def install_nasbench301_configloader_compat() -> None:
    """Patch NASBench-301 ConfigSpace internal-field access."""
    try:
        from ConfigSpace import hyperparameters as CSH
        from nasbench301.surrogate_models import utils as nb301_utils
    except Exception:
        return

    original = getattr(nb301_utils.ConfigLoader, "load_config_space", None)
    if original is None or getattr(original, "_proxydiff_configspace_compat", False):
        return

    def _drop_hyperparameter(config_space, name: str) -> None:
        internal_mapping = getattr(config_space, "_hyperparameters", None)
        if hasattr(internal_mapping, "pop") and internal_mapping is not config_space:
            internal_mapping.pop(name, None)
            return

        dag = getattr(config_space, "_dag", None)
        if dag is None or name not in getattr(dag, "nodes", {}):
            return
        for mapping_name in (
            "nodes",
            "roots",
            "non_roots",
            "index_of",
            "children_of",
            "parents_of",
            "child_conditions_of",
            "parent_conditions_of",
        ):
            mapping = getattr(dag, mapping_name, None)
            if hasattr(mapping, "pop"):
                mapping.pop(name, None)
        dag.at = [hp_name for hp_name in getattr(dag, "at", []) if hp_name != name]
        dag.hyperparameters = [hp for hp in getattr(dag, "hyperparameters", []) if hp.name != name]
        dag.index_of = {hp_name: idx for idx, hp_name in enumerate(dag.at)}
        for idx, hp_name in enumerate(dag.at):
            if hp_name in dag.nodes:
                dag.nodes[hp_name].idx = idx
        config_space._len = len(dag.at)

    @staticmethod
    def _load_config_space_compat(path):
        with open(path, "r") as fh:
            config_space = nb301_utils.config_space_json_r_w.read(fh.read())

        replacements = [
            CSH.UniformIntegerHyperparameter(
                name="NetworkSelectorDatasetInfo:darts:layers", lower=1, upper=10000
            ),
            CSH.UniformIntegerHyperparameter(
                name="SimpleLearningrateSchedulerSelector:cosine_annealing:T_max",
                lower=1,
                upper=10000,
            ),
            CSH.UniformIntegerHyperparameter(
                name="NetworkSelectorDatasetInfo:darts:init_channels", lower=1, upper=10000
            ),
            CSH.UniformFloatHyperparameter(
                name="SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min",
                lower=0,
                upper=10000,
            ),
        ]
        for hp in replacements:
            _drop_hyperparameter(config_space, hp.name)
        config_space.add_hyperparameters(replacements)
        return config_space

    _load_config_space_compat._proxydiff_configspace_compat = True
    nb301_utils.ConfigLoader.load_config_space = _load_config_space_compat


install_nasbench301_configloader_compat()

from _proxydiff_nb301_refinement_impl import (
    ZeroCostPredictorEvaluator,
    install_nb301_canonical_score_forward,
)
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


def enable_deterministic_refinement(seed):
    """Make the refinement feedback loop reproducible when requested."""
    if os.environ.get("PROXYDIFF_DETERMINISTIC_REFINEMENT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def install_fixed_subset_sampler():
    """Optionally decouple CIFAR batch order from global torch RNG state."""
    sampler_seed = os.environ.get("PROXYDIFF_FIXED_SAMPLER_SEED")
    if sampler_seed is None or sampler_seed.strip().lower() in {"", "none", "off", "0"}:
        return

    base_seed = int(sampler_seed)
    original_sampler = torch.utils.data.sampler.SubsetRandomSampler

    class FixedSubsetRandomSampler(original_sampler):
        def __init__(self, indices, generator=None):
            super().__init__(indices, generator=generator)
            self._proxydiff_iter_count = 0

        def __iter__(self):
            generator = torch.Generator()
            generator.manual_seed(base_seed + self._proxydiff_iter_count)
            self._proxydiff_iter_count += 1
            for i in torch.randperm(len(self.indices), generator=generator):
                yield self.indices[i]

    torch.utils.data.sampler.SubsetRandomSampler = FixedSubsetRandomSampler


def apply_evaluator_compat_env():
    """Translate public ProxyDiff names to the bundled evaluator's internal keys."""
    direct = {
        "PROXYDIFF_REFINEMENT_CACHE": evaluator_key("_SCORE_CACHE"),
        "REFINEMENT_AXIS_WEIGHTING": evaluator_key("_METRIC_WEIGHT_STYLE"),
        "AXIS_SCALE_LAYOUT": evaluator_key("_AXIS_CALIB_LAYOUT"),
        "RESIDUAL_AXIS_SCALE": evaluator_key("_RESIDUAL_SCALE"),
        "COMPONENT_CORRECTION_AXIS_SCALE": evaluator_key("_COMPONENT_CORR_SCALE"),
        "AXIS_CALIBRATION_STEPS": evaluator_key("_AXIS_CALIB_STEPS"),
        "TOPK_SELECTION_COUNT": evaluator_key("_PROGRESSIVE_MASK_TOPK"),
        "EVALUATE_INITIAL_SCORE": evaluator_key("_PRETRAIN_EVAL"),
        "INITIAL_SCORE_ONLY": evaluator_key("_PRETRAIN_ONLY"),
        "SAVE_STAGE_ARTIFACTS": evaluator_key("_DUMP_EPOCH_ARTIFACTS"),
        "REFINEMENT_EPOCHS": evaluator_key("_METRIC_EPOCHS"),
        "AXIS_CALIBRATION_LR": evaluator_key("_AXIS_CALIB_LR"),
        "AXIS_CALIBRATION_WEIGHT_DECAY": evaluator_key("_AXIS_CALIB_WEIGHT_DECAY"),
        "COMPONENT_CORRECTION_LR": evaluator_key("_COMPONENT_CORR_LR"),
        "COMPONENT_CORRECTION_WEIGHT_DECAY": evaluator_key("_COMPONENT_CORR_WEIGHT_DECAY"),
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

    topk_policy = os.environ.get("TOPK_SELECTION_POLICY", "").strip().lower()
    if topk_policy:
        if topk_policy == "topk_after_axis_" + "calibration":
            os.environ[evaluator_key("_PROGRESSIVE_MASK")] = "topk_after_axis_calib"
        else:
            os.environ[evaluator_key("_PROGRESSIVE_MASK")] = topk_policy

    topk_source = os.environ.get("TOPK_SELECTION_SOURCE", "").strip().lower()
    if topk_source:
        if topk_source == "axis_calibrated_score":
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = "axis_calib"
        elif topk_source == "axis_calib_with_component_corr":
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = "axis_calib_with_component_corr"
        else:
            os.environ[evaluator_key("_PROGRESSIVE_MASK_SCORE")] = topk_source


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
enable_deterministic_refinement(config.seed)
install_fixed_subset_sampler()

if config.search_space == "nasbench301":
    fixed_arch = os.environ.get("FIXED_ARCH_FILE") or os.environ.get(evaluator_key("_FIXED_ARCH_FILE"))
    if not fixed_arch:
        fixed_arch = str(
            Path(__file__).resolve().parents[1] / "assets" / "arch_dataset_20cell_c36.pt"
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
install_nb301_canonical_score_forward(predictor_evaluator.search_space)
logger.info(
    "ProxyDiff canonical score forward installed for cells=%s",
    predictor_evaluator.search_space._proxydiff_canonical_forward_cells,
)
predictor_evaluator.evaluate(
    zc_api,
    train_epoch=0,
    weight_path=None,
    step=False,
    perturbation=True,
    train_search_weights=False,
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
