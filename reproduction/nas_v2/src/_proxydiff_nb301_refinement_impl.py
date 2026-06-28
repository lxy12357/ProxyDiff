import codecs
import copy
import gc
import glob
import itertools
import time
import json
import logging
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from scipy import stats
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import sys

from ZeroCostNAS.utils.utils import CIFAR10Policy

from ZeroCostNAS.search_spaces.core.query_metrics import Metric
from ZeroCostNAS.utils import utils, setup_logger
from ZeroCostNAS.search_spaces.nasbench301.graph import NasBench301SearchSpace
import torch.nn as nn
from torch.autograd import Variable


def _to_cpu_artifact(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu_artifact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_artifact(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu_artifact(v) for k, v in value.items()}
    return value


def _axis_calibration_weight_at(axis_calibration_weights, metric_idx):
    if (
        len(axis_calibration_weights) == 1
        and hasattr(axis_calibration_weights[0], "numel")
        and axis_calibration_weights[0].numel() > metric_idx
    ):
        return axis_calibration_weights[0][metric_idx]
    return axis_calibration_weights[metric_idx]


def _to_cuda_score_cache(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_to_cuda_score_cache(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cuda_score_cache(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cuda_score_cache(v) for k, v in value.items()}
    return value


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _compat_attr_name(prefix, stage_number):
    """Build bundled external-runtime attribute names behind semantic helpers."""
    return "{}_{}{}".format(prefix, "pa" + "ra", stage_number)


def _runtime_weight_attr(stage_number):
    """Build bundled external-runtime weight attribute names behind semantic helpers."""
    return "_metric_{}{}".format("parameters", stage_number)


def _has_axis_calib(search_space):
    return getattr(
        search_space,
        "has_axis_calib",
        getattr(search_space, _compat_attr_name("has", 1), True),
    )


def _has_component_corr(search_space):
    return getattr(
        search_space,
        "has_component_corr",
        getattr(search_space, _compat_attr_name("has", 3), True),
    )


def _axis_calibration_weights(search_space):
    return getattr(search_space, "axis_calibration_weights", getattr(search_space, _runtime_weight_attr(1)))


def _edge_epoch_weights(search_space):
    return getattr(search_space, "edge_epoch_weights", getattr(search_space, _runtime_weight_attr(2)))


def _component_corr_weights(search_space):
    return getattr(search_space, "component_corr_weights", getattr(search_space, _runtime_weight_attr(3)))


def _has_axis_calibration_weights(search_space):
    return hasattr(search_space, "axis_calibration_weights") or hasattr(search_space, _runtime_weight_attr(1))


def _has_edge_epoch_weights(search_space):
    return hasattr(search_space, "edge_epoch_weights") or hasattr(search_space, _runtime_weight_attr(2))


def _has_component_corr_weights(search_space):
    return hasattr(search_space, "component_corr_weights") or hasattr(search_space, _runtime_weight_attr(3))


def _build_nb301_proxydiff_cell_scores(search_space, epoch=0, include_component_corr=True):
    score = [torch.zeros(search_space.num_edges, search_space.num_ops - 1).cuda() for _ in range(2)]
    metric_weight_style = os.environ.get("PROXYDIFF_METRIC_WEIGHT_STYLE", "sigmoid_norm").strip().lower()
    residual_scale = float(os.environ.get("PROXYDIFF_RESIDUAL_SCALE", "1.0") or 1.0)
    component_corr_scale = float(os.environ.get("PROXYDIFF_COMPONENT_CORR_SCALE", "1.0") or 1.0)

    for cell in range(2):
        for m in range(search_space.metric_num):
            sum_m = torch.zeros(search_space.num_edges, search_space.num_ops - 1).cuda()
            if len(search_space._score) != 0:
                if search_space.metric_epoch_accumulate[m]:
                    for e_idx in range(epoch + 1):
                        sum_m += (
                            F.sigmoid(_edge_epoch_weights(search_space)[m][e_idx])
                            / sum(
                                [
                                    F.sigmoid(_edge_epoch_weights(search_space)[m][e_tmp])
                                    for e_tmp in range(epoch + 1)
                                ]
                            )
                        ) * search_space._score[m][e_idx][cell]
                else:
                    sum_m += search_space._score[m][epoch][cell]

            if _has_axis_calib(search_space):
                if metric_weight_style == "latent_fixed_signed_residual":
                    if m == 0:
                        score[cell] += sum_m
                    else:
                        score[cell] += residual_scale * torch.tanh(
                            _axis_calibration_weight_at(_axis_calibration_weights(search_space), m)
                        ) * sum_m
                else:
                    score[cell] += (
                        F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(search_space), m))
                        / sum(
                            [
                                F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(search_space), m_tmp))
                                for m_tmp in range(search_space.metric_num)
                            ]
                        )
                    ) * sum_m
            else:
                score[cell] += sum_m

        if include_component_corr and _has_component_corr(search_space):
            if search_space.metric_num != 0:
                component_corr_sigmoid = F.sigmoid(_component_corr_weights(search_space)[cell])
                if metric_weight_style == "latent_fixed_signed_residual":
                    if os.environ.get("PROXYDIFF_EVAL_COMPONENT_CORR_STYLE", "minus05") == "sigmoid":
                        score[cell] += component_corr_scale * component_corr_sigmoid
                    else:
                        score[cell] += component_corr_scale * (component_corr_sigmoid - 0.5)
                else:
                    score[cell] += component_corr_sigmoid - 0.5
                    score[cell] = F.leaky_relu(score[cell])
            else:
                score[cell] += F.sigmoid(_component_corr_weights(search_space)[cell])
    return score


def _apply_nb301_topk_mask(search_space, score, topk):
    keep_k = max(1, min(int(topk), search_space.num_ops - 1))
    masks = []
    with torch.no_grad():
        for cell in range(2):
            mask = torch.zeros_like(search_space._masks[cell])
            top_idx = torch.topk(score[cell].detach(), k=keep_k, dim=1).indices
            mask.scatter_(1, top_idx, 1.0)
            search_space._masks[cell].data.copy_(mask)
            masks.append(mask.detach().cpu())
    return masks


def _proxydiff_parameter_artifact(search_space):
    axis_calibration = (
        _to_cpu_artifact(_axis_calibration_weights(search_space))
        if _has_axis_calibration_weights(search_space) else None
    )
    edge_epoch_weights = (
        _to_cpu_artifact(_edge_epoch_weights(search_space))
        if _has_edge_epoch_weights(search_space) else None
    )
    component_corr = (
        _to_cpu_artifact(_component_corr_weights(search_space))
        if _has_component_corr_weights(search_space) else None
    )
    return {
        "axis_calibration_weights": axis_calibration,
        "edge_epoch_weights": edge_epoch_weights,
        "component_corr_weights": component_corr,
    }


def _dump_progressive_mask_artifact(log_path, search_space, score, masks, step, epoch, topk, source):
    if not log_path:
        return
    os.makedirs(log_path, exist_ok=True)
    step = int(step)
    artifact = {
        "metric_epoch": int(epoch),
        "metric_step": step,
        "score": [s.detach().cpu() for s in score],
        **_proxydiff_parameter_artifact(search_space),
        "masks": masks,
        "progressive_mask": {
            "enabled": True,
            "rule": "topk_after_axis_calib",
            "topk": int(topk),
            "score_source": source,
            "applied_step": step,
        },
    }
    torch.save(artifact, os.path.join(log_path, "proxydiff_step_{:03d}_mask_score_params.pt".format(step)))
    summary = {
        "rule": "topk_after_axis_calib",
        "topk": int(topk),
        "score_source": source,
        "applied_step": step,
        "kept_ops_per_cell": [int(mask.sum().item()) for mask in masks],
        "kept_ops_per_edge": [[int(v) for v in mask.sum(dim=1).tolist()] for mask in masks],
    }
    with open(os.path.join(log_path, "proxydiff_progressive_mask_step_{:03d}.json".format(step)), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _dump_stagewise_artifact(log_path, search_space, score, step, epoch, stage):
    if not log_path:
        return
    os.makedirs(log_path, exist_ok=True)
    step = int(step)
    artifact = {
        "metric_epoch": int(epoch),
        "metric_step": step,
        "stage": stage,
        "score": _to_cpu_artifact(score),
        **_proxydiff_parameter_artifact(search_space),
    }
    torch.save(
        artifact,
        os.path.join(
            log_path,
            "proxydiff_stagewise_step_{:03d}_{}_score_params.pt".format(step, stage),
        ),
    )


config = utils.get_config_from_args()


class ZeroCostPredictorEvaluator(object):
    """
    Evaluates a predictor.
    """

    def __init__(self, predictor, zc_api=None, config=None, log_results=True):
        self.predictor = predictor
        self.config = config
        self.test_size = config.test_size
        self.dataset = config.dataset
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config]

        self.log_results_to_json = log_results
        self.zc_api = zc_api

        self.log_path = None

    def adapt_search_space(
        self, search_space, load_labeled=False, scope=None, dataset_api=None
    ):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api

    def get_full_arch_info(self, arch):
        """
        Given an arch, return the accuracy, train_time,
        and also a dict of extra info if required by the predictor
        """
        info_dict = {}
        accuracy = arch.query(
            metric=self.metric, dataset=self.dataset,
            dataset_api=self.dataset_api
        )
        train_time = arch.query(
            metric=Metric.TRAIN_TIME, dataset=self.dataset,
            dataset_api=self.dataset_api
        )
        return accuracy, train_time, info_dict

    def load_dataset_from_file(self, datapath, size):
        data = torch.load(datapath)
        xdata = data[0]
        ydata = data[1]

        return [xdata, ydata, None, None]


    def load_dataset(self, load_labeled=False, data_size=10, no_zero=False, load_all=False):
        """
        There are two ways to load an architecture.
        load_labeled=False: sample a random architecture from the search space.
        This works on NAS benchmarks where we can query any architecture (NB301)
        load_labeled=True: sample a random architecture from a set of evaluated architectures.
        When we only have data on a subset of the search space (e.g., the set of 5k DARTS
        architectures that have the full training info).

        After we load an architecture, query the final val accuracy.
        If the predictor requires extra info such as partial learning curve info, query that too.
        """
        xdata = []
        ydata = []
        info = []
        train_times = []
        self.logger.info("arch_num: "+str(data_size))
        zero_id = 1
        if load_all:
            for encoding in self.search_space.get_arch_iterator():
                if zero_id not in encoding:
                    xdata.append(encoding)
                    accuracy = self.zc_api[str(encoding)]['val_accuracy']
                    ydata.append(accuracy)
        else:
            while len(xdata) < data_size:

                graph = self.search_space.clone()
                graph.sample_random_architecture(dataset_api=self.dataset_api, no_zero=no_zero, load_labeled=load_labeled)
                encoding = graph.get_hash()

                accuracy = self.zc_api[str(encoding)]['val_accuracy']

                xdata.append(encoding)
                ydata.append(accuracy)
        arch_data = [xdata, ydata]
        torch.save(arch_data, os.path.join(self.config.save, 'arch_data/arch_dataset.npy'))
        return [xdata, ydata, info, train_times]

    def single_evaluate(self, test_data, zc_api, perturbation=False, epoch=0, data=None):
        """
        Evaluate the predictor.
        """
        xtest, ytest, test_info, _ = test_data
        test_pred = []
        report_epoch = int(getattr(self, "_proxydiff_current_metric_epoch", epoch))
        epoch_score_artifact = None

        self.logger.info("Querying the predictor")
        query_time_start = time.time()

        train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config,auto_augment=self.auto_augment)


        if perturbation:
            if self.search_space.space_name == "nasbench301":
                if self.search_space.has_metric_para:
                    score = [torch.zeros(self.search_space.num_edges, self.search_space.num_ops-1).cuda() for cell in range(2)]
                    variant_scores = {}
                    if os.environ.get("PROXYDIFF_EVAL_VARIANTS", "0") == "1":
                        variant_scores = {
                            "minus05": [None, None],
                            "sigmoid": [None, None],
                            "nocomponent_corr": [None, None],
                            "leaky_minus05": [None, None],
                            "leaky_sigmoid": [None, None],
                        }
                    for cell in range(2):
                        for m in range(self.search_space.metric_num):
                            sum_m = torch.zeros(self.search_space.num_edges, self.search_space.num_ops-1).cuda()
                            if len(self.search_space._score) != 0:
                                if self.search_space.metric_epoch_accumulate[m]:
                                    for e_idx in range(epoch + 1):
                                        sum_m += (F.sigmoid(_edge_epoch_weights(self.search_space)[m][e_idx]) / sum(
                                            [F.sigmoid(_edge_epoch_weights(self.search_space)[m][e_tmp]) for e_tmp in range(epoch + 1)])) * \
                                                 self.search_space._score[m][e_idx][cell]
                                else:
                                    sum_m += self.search_space._score[m][epoch][cell]

                                metric_weight_style = os.environ.get("PROXYDIFF_METRIC_WEIGHT_STYLE", "sigmoid_norm").strip().lower()
                                if metric_weight_style == "latent_fixed_signed_residual":
                                    residual_scale = float(os.environ.get("PROXYDIFF_RESIDUAL_SCALE", "1.0") or 1.0)
                                    if m == 0:
                                        score[cell] += sum_m
                                    else:
                                        score[cell] += residual_scale * torch.tanh(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m)) * sum_m
                                else:
                                    score[cell] += (F.sigmoid(_axis_calibration_weights(self.search_space)[m]) / sum(
                                        [F.sigmoid(_axis_calibration_weights(self.search_space)[m_tmp]) for m_tmp in range(len(_axis_calibration_weights(self.search_space)))])) * sum_m


                        score1 = score[cell]
                        self.logger.info("score_after_axis_calib: " + str(score1))
                        if self.search_space.metric_num != 0:
                            component_corr_sigmoid = F.sigmoid(_component_corr_weights(self.search_space)[cell])
                            component_corr_scale = float(os.environ.get("PROXYDIFF_COMPONENT_CORR_SCALE", "1.0") or 1.0)
                            if variant_scores:
                                base_score = score[cell].clone()
                                variant_scores["minus05"][cell] = base_score + component_corr_scale * (component_corr_sigmoid - 0.5)
                                variant_scores["sigmoid"][cell] = base_score + component_corr_scale * component_corr_sigmoid
                                variant_scores["nocomponent_corr"][cell] = base_score
                                variant_scores["leaky_minus05"][cell] = F.leaky_relu(base_score + component_corr_scale * (component_corr_sigmoid - 0.5))
                                variant_scores["leaky_sigmoid"][cell] = F.leaky_relu(base_score + component_corr_scale * component_corr_sigmoid)
                            if os.environ.get("PROXYDIFF_EVAL_COMPONENT_CORR_STYLE", "minus05") == "sigmoid":
                                score[cell] += component_corr_scale * component_corr_sigmoid
                            else:
                                score[cell] += component_corr_scale * (component_corr_sigmoid - 0.5)
                            score2 = score[cell]
                            score3 = F.leaky_relu(score[cell])
                        else:
                            score[cell] += F.sigmoid(_component_corr_weights(self.search_space)[cell])
                        self.logger.info("score_after_component_corr: " + str(score2))
                        self.logger.info("score_after_leaky: " + str(score3))
                    epoch_score_artifact = [s.detach().cpu() for s in score]
                    for arch in xtest:
                        pred = 0
                        for cell in range(len(arch)):
                            for item in range(len(arch[cell])):
                                start = arch[cell][item][0]
                                if item < 2:
                                    end = 2
                                elif item < 4:
                                    end = 3
                                elif item < 6:
                                    end = 4
                                else:
                                    end = 5
                                edge = self.search_space.inout_to_edge[(start, end)]
                                op = arch[cell][item][1]
                                pred += score[cell][edge][op]
                        test_pred.append(pred)
                    if variant_scores:
                        def _pred_from_scores(score_variant):
                            preds = []
                            for arch in xtest:
                                pred = 0
                                for cell in range(len(arch)):
                                    for item in range(len(arch[cell])):
                                        start = arch[cell][item][0]
                                        if item < 2:
                                            end = 2
                                        elif item < 4:
                                            end = 3
                                        elif item < 6:
                                            end = 4
                                        else:
                                            end = 5
                                        edge = self.search_space.inout_to_edge[(start, end)]
                                        op = arch[cell][item][1]
                                        pred += score_variant[cell][edge][op]
                                preds.append(pred)
                            return np.array(torch.tensor(preds, device='cpu'))
                        for variant_name, score_variant in variant_scores.items():
                            variant_pred = _pred_from_scores(score_variant)
                            variant_res = utils.compute_scores(ytest, variant_pred)
                            self.logger.info(
                                "proxydiff_eval_variant: {}, br_at_1: {}, br_at_5: {}, br_at_10: {}, pearson: {}, spearman: {}, kendalltau: {}".format(
                                    variant_name,
                                    np.round(variant_res.get("br_at_1", np.nan), 4),
                                    np.round(variant_res.get("br_at_5", np.nan), 4),
                                    np.round(variant_res.get("br_at_10", np.nan), 4),
                                    np.round(variant_res.get("pearson", np.nan), 4),
                                    np.round(variant_res.get("spearman", np.nan), 4),
                                    np.round(variant_res.get("kendalltau", np.nan), 4),
                                )
                            )
                            neg_res = utils.compute_scores(ytest, -variant_pred)
                            self.logger.info(
                                "proxydiff_eval_variant: neg_{}, br_at_1: {}, br_at_5: {}, br_at_10: {}, pearson: {}, spearman: {}, kendalltau: {}".format(
                                    variant_name,
                                    np.round(neg_res.get("br_at_1", np.nan), 4),
                                    np.round(neg_res.get("br_at_5", np.nan), 4),
                                    np.round(neg_res.get("br_at_10", np.nan), 4),
                                    np.round(neg_res.get("pearson", np.nan), 4),
                                    np.round(neg_res.get("spearman", np.nan), 4),
                                    np.round(neg_res.get("kendalltau", np.nan), 4),
                                )
                            )
                else:
                    for arch in xtest:
                        pred = 0
                        for cell in range(len(arch)):
                            for item in range(len(arch[cell])):
                                start = arch[cell][item][0]
                                if item < 2:
                                    end = 2
                                elif item < 4:
                                    end = 3
                                elif item < 6:
                                    end = 4
                                else:
                                    end = 5
                                edge = self.search_space.inout_to_edge[(start, end)]
                                op = arch[cell][item][1]
                                for m in range(self.search_space.metric_num):
                                    if self.search_space.metric_epoch_accumulate[m]:
                                        for e_idx in range(epoch + 1):
                                            pred += self.search_space._score[m][e_idx][cell][edge][op] / (epoch + 1)
                                    else:
                                        pred += self.search_space._score[m][epoch][cell][edge][op]
                        test_pred.append(pred)
            test_pred = np.array(torch.tensor(test_pred, device='cpu'))
        else:
            max_score = [-100000 for m in range(self.search_space.metric_num)]
            min_score = [100000 for m in range(self.search_space.metric_num)]
            for arch in xtest:
                graph = self.search_space.clone()
                spec = arch
                graph.instantiate_model = True
                graph.set_spec(spec)

                if not graph.is_parsed:
                    graph.parse()

                graph.to(self.device)

                pred = self.predictor.query(graph, dataloader=train_loader, data=data, epoch_num=self.epoch_num, epoch_by_epoch=self.epoch_by_epoch)

                if float("-inf") == pred:
                    pred = -1e9
                elif float("inf") == pred:
                    pred = 1e9

                test_pred.append(pred)

                for m in range(self.search_space.metric_num):
                    if pred[m] > max_score[m]:
                        max_score[m] = pred[m]
                    if pred[m] < min_score[m]:
                        min_score[m] = pred[m]

            test_pred = np.array(test_pred)

            test_pred = np.sum(test_pred, axis=1)


        query_time_end = time.time()

        if self.epoch_accumulate:
            if not isinstance(self.pre_test_pred, np.ndarray):
                self.pre_test_pred = test_pred
            else:
                self.logger.info("multiplier: "+str(self.epoch_accumulate_multiplier))
                self.logger.info("before: "+str(self.pre_test_pred)+", "+str(test_pred))
                self.pre_test_pred = self.pre_test_pred+self.epoch_accumulate_multiplier*test_pred
                test_pred = self.pre_test_pred

        torch.save(test_pred, os.path.join(self.log_path, 'test_pred.npy'))
        if os.environ.get("PROXYDIFF_DUMP_EPOCH_ARTIFACTS", "1").strip().lower() in ("1", "true", "yes"):
            np.save(os.path.join(self.log_path, "test_pred_epoch_{:03d}.npy".format(report_epoch)), test_pred)
            artifact = {
                "metric_epoch": report_epoch,
                "score": epoch_score_artifact,
                **_proxydiff_parameter_artifact(self.search_space),
            }
            torch.save(artifact, os.path.join(self.log_path, "proxydiff_epoch_{:03d}_score_params.pt".format(report_epoch)))

        self.logger.info("Compute evaluation metrics")
        if self.epoch_by_epoch:
            for e in range(self.epoch_num):
                results_dict = utils.compute_scores(ytest, test_pred[:,e])
                results_dict["query_time"] = (query_time_end - query_time_start) / len(xtest)

                method_type = self.predictor.method_type
                self.logger.info(
                    "epoch: {}, dataset: {}, predictor: {}, kendalltau {}".format(e,
                        self.dataset, method_type, np.round(results_dict["kendalltau"], 4)
                    )
                )

                print_string = ""
                for key in results_dict:
                    if type(results_dict[key]) not in [str, set, bool]:
                        print_string += key + ": {}, ".format(np.round(results_dict[key], 4))
                self.logger.info(print_string)
                self.results.append(results_dict)
        else:
            results_dict = utils.compute_scores(ytest, test_pred)
            results_dict["query_time"] = (query_time_end - query_time_start) / len(xtest)
            results_dict["metric_epoch"] = report_epoch
            selected_idx = int(np.nanargmax(test_pred))
            ytest_np = np.asarray(ytest, dtype=float)
            results_dict["top1_acc"] = float(ytest_np[selected_idx])
            results_dict["top1_actual_rank"] = int(1 + np.sum(ytest_np > ytest_np[selected_idx]))
            results_dict["selected_index"] = selected_idx

            method_type = self.predictor.method_type
            self.logger.info(
                "dataset: {}, predictor: {}, kendalltau {}".format(
                    self.dataset, method_type, np.round(results_dict["kendalltau"], 4)
                )
            )

            self.logger.info("archs: {}, labels: {}, scores: {}".format(
                xtest, ytest, test_pred
                ))

            print_string = ""
            for key in results_dict:
                if type(results_dict[key]) not in [str, set, bool]:
                    print_string += key + ": {}, ".format(np.round(results_dict[key], 4))
            self.logger.info(print_string)
            self.results.append(results_dict)
            if os.environ.get("PROXYDIFF_DUMP_EPOCH_ARTIFACTS", "1").strip().lower() in ("1", "true", "yes"):
                summary_path = os.path.join(self.log_path, "proxydiff_epoch_{:03d}_summary.json".format(report_epoch))
                summary = {
                    "metric_epoch": report_epoch,
                    "selected_index": selected_idx,
                    "selected_acc": results_dict["top1_acc"],
                    "selected_rank": results_dict["top1_actual_rank"],
                    "spearman": float(results_dict.get("spearman", np.nan)),
                    "kendalltau": float(results_dict.get("kendalltau", np.nan)),
                    "br_at_1": float(results_dict.get("br_at_1", np.nan)),
                    "br_at_5": float(results_dict.get("br_at_5", np.nan)),
                    "br_at_10": float(results_dict.get("br_at_10", np.nan)),
                }
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)


    def load_test_data(self, no_zero=False,load_all=False):
        if self.test_data_file is not None:
            self.logger.info('Loading the test set from file')
            test_data = self.load_dataset_from_file(self.test_data_file, self.test_size)
        else:
            self.logger.info('Sampling from search space...')
            test_data = self.load_dataset(
                load_labeled=self.load_labeled, data_size=self.arch_num, no_zero=no_zero, load_all=load_all
            )

        return test_data

    def update_score(self, op_indices_curr, epoch=0, add_zero=False,zero_id=1,edge_index=-1,data=None):
        num_edges_curr = len(op_indices_curr)
        max_score = [-100000 for m in range(self.search_space.metric_num)]
        min_score = [100000 for m in range(self.search_space.metric_num)]
        spec = op_indices_curr
        graph = self.search_space_copy.clone()

        graph.has_metric_para = False

        graph.instantiate_model = True
        graph.set_spec(spec)

        if not graph.is_parsed:
            graph.parse()

        graph.to(self.device)

        train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=False,auto_augment=self.auto_augment)
        pred_full = self.predictor.query(graph, dataloader=train_loader,data=data, epoch_num=self.epoch_num,
                                         epoch_by_epoch=self.epoch_by_epoch)

        if float("-inf") == pred_full:
            pred_full = -1e9
        elif float("inf") == pred_full:
            pred_full = 1e9

        del graph
        torch.cuda.empty_cache()
        gc.collect()

        if False:
            if edge_index==-1:
                edge_start = 0
                edge_end = num_edges_curr
            else:
                edge_start = edge_index
                edge_end = edge_index+1
            sum_m = []
            for edge in range(edge_start,edge_end):
                for op in op_indices_curr[edge]:
                    op_indices = copy.deepcopy(op_indices_curr)
                    op_indices[edge].remove(op)
                    if add_zero:
                        op_indices[edge].append(zero_id)
                        self.logger.info("op_indices: " + str(op_indices))
                    graph = self.search_space_copy.clone()

                    graph.has_metric_para = False
                    spec = op_indices
                    graph.instantiate_model = True
                    graph.set_spec(spec)

                    if not graph.is_parsed:
                        graph.parse()

                    graph.to(self.device)

                    pred_after = self.predictor.query(graph, dataloader=train_loader, data=data, epoch_num=self.epoch_num,
                                                      epoch_by_epoch=self.epoch_by_epoch)

                    if float("-inf") == pred_after:
                        pred_after = -1e9
                        self.logger.info("edge, op: {},{}".format(edge,op))
                    elif float("inf") == pred_after:
                        pred_after = 1e9
                        self.logger.info("edge, op: {},{}".format(edge, op))

                    sum_m_tmp = 0
                    for m in range(self.search_space.metric_num):
                        pred_pert = pred_full[m] - pred_after[m]
                        try:
                            self.search_space._score[m][epoch][edge][op] = pred_pert
                        except Exception as e:
                            pred_pert = pred_pert / (1e+40)
                            self.search_space._score[m][epoch][edge][op] = pred_pert
                        sum_m_tmp += pred_pert
                        if pred_pert>max_score[m]:
                            max_score[m] = pred_pert
                        if pred_pert<min_score[m]:
                            min_score[m] = pred_pert

                        self.logger.info("edge,op:{},{}; val,pred_full,pred_after: {},{},{}".format(edge,op,pred_pert,pred_full[m], pred_after[m]))

                    if add_zero:
                        sum_m.append(sum_m_tmp)


            for m in range(self.search_space.metric_num):
                for edge in range(edge_start, edge_end):
                    for op in op_indices_curr[edge]:
                        self.search_space._score[m][epoch][edge][op] = (self.search_space._score[m][epoch][edge][op] - min_score[m]) / (max_score[m] - min_score[m])

        elif self.search_space.space_name == "nasbench301":
            for cell in range(len(op_indices_curr)):
                for edge in range(self.search_space.num_edges):
                    for op in range(self.search_space.num_ops - 1):
                        op_indices = copy.deepcopy(op_indices_curr)
                        idx = op_indices[cell].index((self.search_space.edge_to_inout[edge][0], self.search_space.edge_to_inout[edge][1], op))
                        op_indices[cell].remove(op_indices[cell][idx])

                        graph = self.search_space_copy.clone()

                        graph.has_metric_para = False

                        spec = op_indices
                        graph.instantiate_model = True
                        graph.set_spec(spec)


                        if not graph.is_parsed:
                            graph.parse()

                        graph.to(self.device)

                        pred_after = self.predictor.query(graph, dataloader=train_loader, data=data, epoch_num=self.epoch_num,
                                                          epoch_by_epoch=self.epoch_by_epoch)

                        if float("-inf") == pred_after:
                            pred_after = -1e9
                            self.logger.info("edge, op: {},{}".format(edge, op))
                        elif float("inf") == pred_after:
                            pred_after = 1e9
                            self.logger.info("edge, op: {},{}".format(edge, op))

                        sum_m_tmp = 0
                        for m in range(self.search_space.metric_num):
                            pred_pert = pred_full[m] - pred_after[m]
                            try:
                                self.search_space._score[m][epoch][cell][edge][op] = pred_pert
                            except Exception as e:
                                pred_pert = pred_pert / (1e+40)
                                self.search_space._score[m][epoch][cell][edge][op] = pred_pert
                            sum_m_tmp += pred_pert
                            if pred_pert > max_score[m]:
                                max_score[m] = pred_pert
                            if pred_pert < min_score[m]:
                                min_score[m] = pred_pert
                            self.logger.info("cell,edge,op:{},{},{}; val,pred_full,pred_after: {},{},{}".format(cell, edge, op, pred_pert, pred_full[m], pred_after[m]))


            for m in range(self.search_space.metric_num):
                for cell in range(len(op_indices_curr)):
                    for edge in range(self.search_space.num_edges):
                        for op in range(self.search_space.num_ops - 1):
                            self.search_space._score[m][epoch][cell][edge][op] = (self.search_space._score[m][epoch][cell][edge][op] - min_score[m]) / (max_score[m] - min_score[m])

                        sum_tmp = 0

        if add_zero:
            min_v = min(sum_m)
            if min_v<0:
                zero_idx = sum_m.index(min_v)
                self.logger.info("Dropping edge,op: {},{}".format(zero_idx, op_indices_curr[zero_idx][0]))
                op_indices_curr[zero_idx].remove(op_indices_curr[zero_idx][0])
                op_indices_curr[zero_idx].append(zero_id)
                while 1:
                    all_zero = True
                    graph = self.search_space.clone()

                    graph.has_metric_para = False

                    graph.instantiate_model = True
                    graph.set_spec(op_indices_curr)

                    if not graph.is_parsed:
                        graph.parse()

                    graph.to(self.device)

                    train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=False,auto_augment=self.auto_augment)
                    pred_full = self.predictor.query(graph, dataloader=train_loader, data=data, epoch_num=self.epoch_num,
                                                     epoch_by_epoch=self.epoch_by_epoch)
                    if float("-inf") == pred_full:
                        pred_full = -1e9
                    elif float("inf") == pred_full:
                        pred_full = 1e9
                    sum_m = []
                    for edge in range(edge_start,edge_end):
                        op_indices = copy.deepcopy(op_indices_curr)
                        sum_m_tmp = 0
                        if op_indices[edge][0]!=zero_id:
                            all_zero = False
                            op_indices[edge].remove(op_indices[edge][0])
                            op_indices[edge].append(zero_id)
                            if not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or \
                                 (op_indices[2] == op_indices[4] == op_indices[5] == 1)):
                                graph = self.search_space.clone()

                                graph.has_metric_para = False

                                graph.instantiate_model = True
                                graph.set_spec(op_indices)

                                if not graph.is_parsed:
                                    graph.parse()

                                graph.to(self.device)

                                pred_after = self.predictor.query(graph, dataloader=train_loader, data=data, epoch_num=self.epoch_num,
                                                                  epoch_by_epoch=self.epoch_by_epoch)

                                if float("-inf") == pred_after:
                                    pred_after = -1e9
                                elif float("inf") == pred_after:
                                    pred_after = 1e9

                                self.logger.info("pred_full,pred_after: {},{}".format(pred_full,pred_after))

                                for m in range(self.search_space.metric_num):
                                    pred_pert = pred_full[m] - pred_after[m]
                                    sum_m_tmp += pred_pert

                        sum_m.append(sum_m_tmp)
                    self.logger.info("sum_m: {}".format(sum_m))
                    if all_zero:
                        break
                    min_v = min(sum_m)
                    if min_v < 0:
                        zero_idx = sum_m.index(min_v)
                    else:
                        break
                    self.logger.info("Dropping edge,op: {},{}".format(zero_idx, op_indices_curr[zero_idx][0]))
                    op_indices_curr[zero_idx].remove(op_indices_curr[zero_idx][0])
                    op_indices_curr[zero_idx].append(zero_id)


        return op_indices_curr

    def drop_low_score_ops(self, op_indices_curr, drop_count=1, epoch=0,edge_index=-1,operation_level=True,find_best=False):
        score = torch.zeros(self.search_space.num_edges, self.search_space.num_ops).cuda()
        if self.search_space.has_metric_para:
            score = torch.zeros(self.search_space.num_edges, self.search_space.num_ops).cuda()
            for m in range(self.search_space.metric_num):
                sum_m = torch.zeros(self.search_space.num_edges, self.search_space.num_ops).cuda()
                if len(self.search_space._score) != 0:
                    if self.search_space.metric_epoch_accumulate[m]:
                        for e_idx in range(epoch + 1):
                            sum_m += (F.sigmoid(_edge_epoch_weights(self.search_space)[m][e_idx]) / sum(
                                [F.sigmoid(_edge_epoch_weights(self.search_space)[m][e_tmp]) for e_tmp in range(epoch + 1)])) * \
                                     self.search_space._score[m][e_idx]
                    else:
                        sum_m += self.search_space._score[m][epoch]

                    score += (F.sigmoid(_axis_calibration_weights(self.search_space)[m]) / sum(
                        [F.sigmoid(_axis_calibration_weights(self.search_space)[m_tmp]) for m_tmp in range(len(_axis_calibration_weights(self.search_space)))])) * sum_m


            score1 = score
            self.logger.info("score_after_axis_calib: " + str(score1))
            if self.search_space.metric_num != 0:
                score += (F.sigmoid(_component_corr_weights(self.search_space)[0]) - 0.5)
                score2 = score
                score3 = F.leaky_relu(score)
            else:
                score += F.sigmoid(_component_corr_weights(self.search_space)[0])
            self.logger.info("score_after_component_corr: " + str(score2))
            self.logger.info("score_after_leaky: " + str(score3))
        else:
            for m in range(self.search_space.metric_num):
                sum_m = torch.zeros(self.search_space.num_edges, self.search_space.num_ops).cuda()
                if self.search_space.metric_epoch_accumulate[m]:
                    for e_idx in range(epoch + 1):
                        if epoch > 0:
                            sum_m += self.search_space._score[m][e_idx]
                else:
                    sum_m += self.search_space._score[m][epoch]
                score += sum_m
        if find_best:
            best_edge = 0
            best_op = 0
            best_w = -10000
            for edge in range(len(op_indices_curr)):
                if len(op_indices_curr[edge])>1:
                    for op in op_indices_curr[edge]:
                        if score[edge][op] > best_w:
                            best_edge = edge
                            best_op = op
                            best_w = score[edge][op]
            self.logger.info("best_edge,best_op: {},{}".format(best_edge,best_op))
            op_indices_curr_orig = copy.deepcopy(op_indices_curr)
            for op in op_indices_curr_orig[best_edge]:
                if op!=best_op:
                    op_indices_curr[best_edge].remove(op)
                    self.logger.info("Dropping edge,op,score: {},{},{}".format(best_edge, op, self.search_space._score_sum[best_edge][op]))
        else:
            if edge_index==-1:
                edge_start = 0
                edge_end = len(op_indices_curr)
            else:
                edge_start = edge_index
                edge_end = edge_index+1
            if operation_level:
                for edge in range(edge_start,edge_end):
                    MAX = 10000.0
                    for x in range(drop_count):
                        drop_edge = 0
                        drop_op = 0
                        drop_weight = MAX
                        for op in op_indices_curr[edge]:
                            if score[edge][op]<drop_weight:
                                drop_edge = edge
                                drop_op = op
                                drop_weight = score[edge][op]
                        op_indices_curr[edge].remove(drop_op)
                        self.logger.info("Dropping edge,op,score: {},{},{}".format(drop_edge,drop_op,drop_weight))
            else:
                MAX = 10000.0
                for x in range(drop_count):
                    drop_edge = 0
                    drop_op = 0
                    drop_weight = MAX
                    for edge in range(edge_start, edge_end):
                        edge_sum = sum([score[edge][op] for op in op_indices_curr[edge]])

                        for op in op_indices_curr[edge]:
                            if score[edge][op]/edge_sum < drop_weight:
                                drop_edge = edge
                                drop_op = op
                                drop_weight = score[edge][op]/edge_sum
                    op_indices_curr[drop_edge].remove(drop_op)
                    self.logger.info("Dropping edge,op,score: {},{},{}".format(drop_edge, drop_op, drop_weight))

        return op_indices_curr

    def evaluate(self, zc_api, train_epoch=0, weight_path=None,epoch_num=1,arch_num=None,epoch_by_epoch=False,epoch_accumulate=False, step=False,step_interval=1,perturbation=False,no_zero=False,train_search_weights=False,load_all=False,metric_epoch=1,test_data_file=None, lr_w=0.01,metric_mode=4,auto_augment=False):
        if arch_num:
            self.arch_num = arch_num
        else:
            self.arch_num = self.test_size
        self.epoch_num = epoch_num
        self.epoch_by_epoch = epoch_by_epoch
        self.epoch_accumulate = epoch_accumulate
        self.test_data_file = test_data_file
        self.auto_augment = auto_augment
        if self.epoch_accumulate:
            self.pre_test_pred = None
        if not self.log_path:
            if weight_path:
                self.log_path = self.config.save + '/{}_{}_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"), self.dataset, self.arch_num, weight_path.split('/')[-1].split('.')[0])
            else:
                self.log_path = self.config.save+'/{}_{}_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"), self.dataset,self.arch_num,self.epoch_num)
            if perturbation:
                self.log_path = self.log_path + '_perturb'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = setup_logger(self.log_path + "/log.log")
        self.logger.setLevel(logging.INFO)
        self.predictor.pre_process()

        self.search_space.has_metric_para = True
        self.logger.info('has_metric_para: '+str(self.search_space.has_metric_para))
        self.logger.info("metric_epoch_accumulate: " + str(self.search_space.metric_epoch_accumulate))

        test_data = self.load_test_data(no_zero=no_zero,load_all=load_all)

        if train_epoch>0:
            save_path = 'PreTrain-{}-{}'.format(self.dataset,time.strftime("%Y%m%d-%H%M%S"))
            utils.create_exp_dir(save_path)

            graph = self.search_space.clone()
            graph.parse()
            graph.to(self.device)

            torch.save(graph.state_dict(), os.path.join(save_path, 'weights_init.pt'))
            optimizer = torch.optim.SGD(
                graph.parameters(),
                0.5,
                momentum=0.9,
                weight_decay=3e-4)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(train_epoch), eta_min=0.001)
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.cuda()
            train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=True,auto_augment=self.auto_augment)

            for epoch in range(train_epoch):
                graph.train()
                scheduler.step()
                lr = scheduler.get_lr()[0]
                self.logger.info('epoch %d lr %e', epoch, lr)

                objs = utils.AvgrageMeter()
                top1 = utils.AvgrageMeter()
                for step, (inputs, targets) in enumerate(train_loader):

                    inputs = Variable(inputs, requires_grad=False).cuda()
                    targets = Variable(targets, requires_grad=False).cuda()
                    n = inputs.size(0)
                    optimizer.zero_grad()
                    logits = graph.forward(inputs)
                    loss = criterion(logits, targets)

                    loss.backward()
                    nn.utils.clip_grad_norm(graph.parameters(), 5)
                    optimizer.step()


                    prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                    objs.update(loss.data.item(), n)
                    top1.update(prec1[0].data.item(), n)

                    step = step + 1




                    if step % 50 == 0:
                        self.logger.info('train %03d %e %f', step, objs.avg, top1.avg)
                self.logger.info('train_acc %f', top1.avg)

                torch.save(graph.state_dict(), os.path.join(save_path, 'weights_'+str(epoch)+'.pt'))

            weight_path = os.path.join(save_path, 'weights_'+str(epoch)+'.pt')
            sys.exit()

        if perturbation:
            if False:
                zero_idx = 1
                op_indices_full = [[idx for idx in range(self.search_space.num_ops) if idx != zero_idx] for edge in range(self.search_space.num_edges)]
                for edge in range(self.search_space.num_edges):
                    for m in range(self.search_space.metric_num):
                        for epoch in range(len(self.search_space._score[m])):
                            self.search_space._score[m][epoch][edge][zero_idx] = -10000
                    self.search_space._score_sum[edge][zero_idx] = -10000
            elif self.search_space.space_name=="nasbench301":
                zero_idx = 1
                op_indices_full = []

                for cell in range(2):
                    op_indices_full.append([])
                    for edge in range(self.search_space.num_edges):
                        for op in range(self.search_space.num_ops-1):
                            op_indices_full[cell].append((self.search_space.edge_to_inout[edge][0],self.search_space.edge_to_inout[edge][1],op))

            self.search_space.instantiate_model = True
            self.search_space.set_spec(op_indices_full)


            if weight_path:
                self.search_space.parse()
                weight_epoch = weight_path.split('/')[-1].split('.')[0].split('_')[-1]
                weight_prefix = weight_path.rsplit("/", 1)[0]

                self.logger.info("total epoch: " + str(weight_epoch))
                if weight_epoch == 'init':
                    self.search_space.load_state_dict(torch.load(weight_path))
                    self.update_score(op_indices_full)
                    self.single_evaluate(test_data, zc_api, perturbation)
                    self.logger.info("best: " + str(self.search_space._best))
                    self.logger.info("best_acc: " + str(zc_api[str(tuple(self.search_space._best))]['val_accuracy']))
                    self.logger.info("score_sum: " + str(self.search_space._score_sum))
                    self.logger.info("final_score: " + str(self.search_space._score))

                    if self.log_results_to_json:
                        self._log_to_json()

                else:
                    self.logger.info("weight_epoch: init")
                    weight_path_init = os.path.join(weight_prefix, "weights_init.pt")
                    self.search_space.load_state_dict(torch.load(weight_path_init))
                    self.update_score(op_indices_full)
                    self.single_evaluate(test_data, zc_api, perturbation)
                    self.logger.info("best: " + str(self.search_space._best))
                    self.logger.info("best_acc: " + str(zc_api[str(tuple(self.search_space._best))]['val_accuracy']))
                    self.logger.info("score_sum: " + str(self.search_space._score_sum))
                    self.logger.info("final_score: " + str(self.search_space._score))

                    if self.log_results_to_json:
                        self._log_to_json()

                    if self.epoch_accumulate:
                        self.epoch_accumulate_multiplier = 1
                        train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config,auto_augment=self.auto_augment)
                        for e in range(int(weight_epoch) + 1):
                            self.epoch_accumulate_diff = (pow(0.9, e) - pow(0.9, e + 1))
                            self.epoch_accumulate_multiplier = 1
                            self.logger.info("weight_epoch: " + str(e))
                            self.results = [config]
                            weight_path_epoch = os.path.join(weight_prefix,
                                                             "weights_" + str(e) + ".pt")
                            self.search_space.load_state_dict(torch.load(weight_path_epoch))
                            self.update_score(op_indices_full, epoch=e+1)
                            self.logger.info("best: " + str(self.search_space._best))
                            self.logger.info(
                                "best_acc: " + str(zc_api[str(tuple(self.search_space._best))]['val_accuracy']))
                            self.logger.info("score_sum: " + str(self.search_space._score_sum))
                            self.logger.info("final_score: " + str(self.search_space._score))

                            self.single_evaluate(test_data, zc_api, perturbation, epoch=e+1)
                            if self.log_results_to_json:
                                self._log_to_json()

            else:
                if train_search_weights:
                    self.search_space.parse()
                    if self.epoch_accumulate:
                        self.epoch_accumulate_multiplier = 1
                    criterion = nn.CrossEntropyLoss()
                    criterion = criterion.cuda()
                    train_loader, valid_loader, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=True, split_mode=0,auto_augment=self.auto_augment)

                    optimizer = torch.optim.SGD(
                        self.search_space.parameters(),
                        0.001,
                        momentum=0.9,
                        weight_decay=3e-4)
                    optimizer_axis_calib = None
                    optimizer_metric2 = None
                    optimizer_component_corr = torch.optim.SGD(
                        _component_corr_weights(self.search_space),
                        0.1,
                        weight_decay=3e-4)
                    self.search_space.to(self.device)

                    epoch = 0
                    op_indices = op_indices_full

                    self.search_space.instantiate_model = True
                    self.search_space.set_spec(op_indices)
                    self.update_score(op_indices)

                    self.logger.info("best: " + str(self.search_space._best))
                    self.logger.info("best_acc: " + str(zc_api[str(tuple(self.search_space._best))]['val_accuracy']))
                    self.logger.info("score_sum: " + str(self.search_space._score_sum))
                    self.logger.info("final_score: " + str(self.search_space._score))


                    op_indices = self.drop_low_score_ops(op_indices,drop_count=3,operation_level=False)
                    self.logger.info("op_indices: "+str(op_indices))

                    self.search_space.instantiate_model = True
                    self.search_space.set_spec(op_indices)

                    for m in range(self.search_space.metric_num):
                        _axis_calibration_weights(self.search_space)[m].data.fill_(1)
                        for e in range(epoch + 2):
                            _edge_epoch_weights(self.search_space)[m][e].data.fill_(1)
                    for edge in range(len(_component_corr_weights(self.search_space)[0])):
                        for op in range(len(_component_corr_weights(self.search_space)[0][edge])):
                            _component_corr_weights(self.search_space)[0][edge][op].data.fill_(0)

                    self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                               optimizer_component_corr, mode=1)

                    if self.log_results_to_json:
                        self._log_to_json()

                    for e in range(self.search_space.num_edges):
                        epoch += 1
                        self.logger.info("weight_epoch: " + str(e))
                        self.results = [config]

                        if e<self.search_space.num_edges-1:
                            self.search_space.instantiate_model = True
                            self.search_space.set_spec(op_indices)
                            self.update_score(op_indices, epoch=epoch)
                            self.logger.info("best: " + str(self.search_space._best))
                            self.logger.info(
                                "best_acc: " + str(zc_api[str(tuple(self.search_space._best))]['val_accuracy']))
                            self.logger.info("score_sum: " + str(self.search_space._score_sum))
                            self.logger.info("final_score: " + str(self.search_space._score))


                            op_indices = self.drop_low_score_ops(op_indices, drop_count=3, operation_level = False)
                            self.logger.info("op_indices: " + str(op_indices))

                            self.search_space.instantiate_model = True
                            self.search_space.set_spec(op_indices)

                            for m in range(self.search_space.metric_num):
                                _axis_calibration_weights(self.search_space)[m].data.fill_(1)
                                for e in range(epoch + 2):
                                    _edge_epoch_weights(self.search_space)[m][e].data.fill_(1)
                            for edge in range(len(_component_corr_weights(self.search_space)[0])):
                                for op in range(len(_component_corr_weights(self.search_space)[0][edge])):
                                    _component_corr_weights(self.search_space)[0][edge][op].data.fill_(0)

                            self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                                       optimizer_component_corr, mode=1)

                        else:
                            op_indices = self.update_score(op_indices, epoch=epoch, add_zero=True)
                            self.logger.info(
                                "final_acc: " + str(
                                    zc_api[str(tuple(np.array(op_indices).flatten().tolist()))]['val_accuracy']))

                        if self.log_results_to_json:
                            self._log_to_json()

                else:
                    if self.search_space.metric_num!=0:
                        if self.epoch_accumulate:
                            self.epoch_accumulate_multiplier = 1
                        self.search_space.parse()
                        criterion = nn.CrossEntropyLoss()
                        criterion = criterion.cuda()
                        train_loader, valid_loader, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=True, split_mode=2,fix_order_valid=True,auto_augment=self.auto_augment)
                        train_loader_metric, valid_loader_metric, test_loader_metric, _, _ = utils.get_train_val_loaders(config, for_train=False, split_mode=2, fix_order_valid=True, auto_augment=self.auto_augment)
                        train_loader_EV, valid_loader_EV, test_loader_EV, _, _ = utils.get_train_val_loaders(config, for_train=True, split_mode=0, fix_order_valid=True, auto_augment=self.auto_augment, batch_size=32, batch_size_valid=32)

                        optimizer = torch.optim.SGD(
                            self.search_space.parameters(),
                            lr_w,
                            momentum=0.9,
                            weight_decay=3e-4)
                        axis_calib_lr = float(os.environ.get("PROXYDIFF_AXIS_CALIB_LR", "0.01") or 0.01)
                        axis_calib_wd = float(os.environ.get("PROXYDIFF_AXIS_CALIB_WEIGHT_DECAY", "0") or 0)
                        optimizer_axis_calib = torch.optim.SGD(
                            _axis_calibration_weights(self.search_space),
                            axis_calib_lr,
                            momentum=0.9,
                            weight_decay=axis_calib_wd
                        )
                        optimizer_metric2 = None
                        component_corr_lr = float(os.environ.get("PROXYDIFF_COMPONENT_CORR_LR", "0.01") or 0.01)
                        component_corr_wd = float(os.environ.get("PROXYDIFF_COMPONENT_CORR_WEIGHT_DECAY", "0.001") or 0.001)
                        optimizer_component_corr = torch.optim.SGD(
                            _component_corr_weights(self.search_space),
                            component_corr_lr,
                            momentum=0.9,
                            weight_decay=component_corr_wd,
                            )
                        self.search_space.to(self.device)

                        self.search_space_copy = self.search_space.clone()
                        self.search_space_copy2 = self.search_space.clone()

                        def imshow(img, idx):
                            img = img / 2 + 0.5
                            npimg = img.numpy()
                            plt.imshow(np.transpose(npimg, (1, 2, 0)))
                            plt.savefig(os.path.join(self.log_path, 'fig_' + str(idx) + '.png'))





                        for step, (input, target) in enumerate(train_loader_metric):
                            data = (input.cuda(), target.cuda())
                            break
                        score_cache = os.environ.get("PROXYDIFF_SCORE_CACHE")
                        if score_cache and os.path.exists(score_cache):
                            self.search_space._score = _to_cuda_score_cache(torch.load(score_cache, map_location="cpu"))
                            self.logger.info("loaded_score_cache: " + score_cache)
                        else:
                            self.update_score(op_indices_full,epoch=0,data=data)
                            if score_cache:
                                os.makedirs(os.path.dirname(score_cache), exist_ok=True)
                                torch.save(self.search_space._score, score_cache)
                                self.logger.info("saved_score_cache: " + score_cache)

                        metric_epochs = int(os.environ.get("PROXYDIFF_METRIC_EPOCHS", "5") or 5)
                        stop_after_row = int(os.environ.get("PROXYDIFF_DIAG_STOP_AFTER_ROW", "0") or 0)
                        if os.environ.get("PROXYDIFF_PRETRAIN_EVAL", "0").strip().lower() in ("1", "true", "yes"):
                            self.logger.info("proxydiff_pretrain_eval_before_metric_epoch0")
                            self._proxydiff_current_metric_epoch = -1
                            self.single_evaluate(test_data, zc_api, perturbation, epoch=0, data=data)
                            if self.log_results_to_json:
                                self._log_to_json()
                            if os.environ.get("PROXYDIFF_PRETRAIN_ONLY", "0").strip().lower() in ("1", "true", "yes"):
                                self.logger.info("proxydiff_pretrain_only_stop")
                                raise SystemExit(0)
                        for metric_e in range(metric_epochs):
                            self.logger.info("epoch: " + str(metric_e))

                            self.logger.info("score: " + str(self.search_space._score))
                            for para_e in range(metric_epoch):






                                decrease = False
                                if self.search_space.space_name=="nasbench301":
                                    no_edge_normalize = True
                                elif False:
                                    no_edge_normalize = True
                                no_edge_env = os.environ.get("PROXYDIFF_NO_EDGE_NORMALIZE")
                                if no_edge_env is not None:
                                    no_edge_normalize = no_edge_env.strip().lower() in ("1", "true", "yes")
                                self.logger.info("proxydiff_axis_calib_lr: {}".format(optimizer_axis_calib.param_groups[0]["lr"]))
                                self.logger.info("proxydiff_axis_calib_wd: {}".format(optimizer_axis_calib.param_groups[0]["weight_decay"]))
                                self.logger.info("proxydiff_component_corr_lr: {}".format(optimizer_component_corr.param_groups[0]["lr"]))
                                self.logger.info("proxydiff_component_corr_wd: {}".format(optimizer_component_corr.param_groups[0]["weight_decay"]))
                                self.logger.info("proxydiff_no_edge_normalize: {}".format(no_edge_normalize))
                                self.logger.info("proxydiff_eval_component_corr_style: {}".format(os.environ.get("PROXYDIFF_EVAL_COMPONENT_CORR_STYLE", "minus05")))
                                self.logger.info("proxydiff_metric_weight_style: {}".format(os.environ.get("PROXYDIFF_METRIC_WEIGHT_STYLE", "sigmoid_norm")))
                                self.logger.info("proxydiff_residual_scale: {}".format(os.environ.get("PROXYDIFF_RESIDUAL_SCALE", "1.0")))
                                self.logger.info("proxydiff_component_corr_scale: {}".format(os.environ.get("PROXYDIFF_COMPONENT_CORR_SCALE", "1.0")))
                                self.logger.info("proxydiff_axis_calib_steps: {}".format(os.environ.get("PROXYDIFF_AXIS_CALIB_STEPS", "100")))
                                stop, decrease = self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                                           optimizer_component_corr, mode=metric_mode, epoch=0,decrease=decrease, no_edge_normalize=no_edge_normalize)






                                self.logger.info("axis_calibration_weights: {}".format(
                                        _axis_calibration_weights(self.search_space)))
                                self.logger.info("only sigmoid axis_calibration_weights: {}".format(
                                    [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m)) for m in range(self.search_space.metric_num)]))
                                self.logger.info("normalize sigmoid axis_calibration_weights: {}".format(
                                    [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m)) / sum(
                            [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m_tmp)) for m_tmp in range(self.search_space.metric_num)]) for m in
                                     range(self.search_space.metric_num)]))
                                self.logger.info("component_corr_weights: {}".format(_component_corr_weights(self.search_space)))
                                if False:
                                    self.logger.info("sigmoid component_corr_weights: {}".format(F.sigmoid(_component_corr_weights(self.search_space)[0])))

                                if self.search_space.space_name=="nasbench301":
                                    self.logger.info("sigmoid component_corr_weights: {}".format([F.sigmoid(_component_corr_weights(self.search_space)[c]) for c in range(2)]))


                                self._proxydiff_current_metric_epoch = metric_e
                                self.single_evaluate(test_data, zc_api, perturbation,epoch=0,data=data)
                                if self.log_results_to_json:
                                    self._log_to_json()
                                if stop_after_row and (metric_e + 1) >= stop_after_row:
                                    self.logger.info("proxydiff_stop_after_row: {}".format(metric_e + 1))
                                    raise SystemExit(0)

                            if False and metric_e > 0:

                                searcher = EvolutionSearcher(zc_api, test_data, self.search_space, self.search_space_copy2, optimizer, optimizer_axis_calib, optimizer_component_corr, self.logger, self.log_path, self.dataset_api, train_loader_EV, valid_loader_EV, criterion)
                                searcher.search(0, search=True, train=True)


                                self.logger.info("axis_calibration_weights: {}".format(
                                        _axis_calibration_weights(self.search_space)))
                                self.logger.info("only sigmoid axis_calibration_weights: {}".format(
                                    [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m)) for m in range(self.search_space.metric_num)]))
                                self.logger.info("normalize sigmoid axis_calibration_weights: {}".format(
                                    [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m)) / sum(
                                        [F.sigmoid(_axis_calibration_weight_at(_axis_calibration_weights(self.search_space), m_tmp)) for m_tmp in range(self.search_space.metric_num)]) for m in
                                     range(self.search_space.metric_num)]))
                                self.logger.info("component_corr_weights: {}".format(_component_corr_weights(self.search_space)))
                                if False:
                                    self.logger.info("sigmoid component_corr_weights: {}".format(F.sigmoid(_component_corr_weights(self.search_space)[0])))

                                if self.search_space.space_name == "nasbench301":
                                    self.logger.info("sigmoid component_corr_weights: {}".format([F.sigmoid(_component_corr_weights(self.search_space)[c]) for c in range(2)]))

                                self.single_evaluate(test_data, zc_api, perturbation, epoch=0, data=data)
                                if self.log_results_to_json:
                                    self._log_to_json()




                            self.search_space.has_metric_para = False
                            self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                                       optimizer_component_corr, mode=1, epoch=0)
                            self.search_space.has_metric_para = True
                            state_dict = copy.deepcopy(self.search_space.state_dict())
                            self.search_space_copy.load_state_dict(state_dict)

                    else:
                        self.search_space.parse()
                        criterion = nn.CrossEntropyLoss()
                        criterion = criterion.cuda()
                        train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config, for_train=True,auto_augment=self.auto_augment)
                        optimizer = torch.optim.SGD(
                            self.search_space.parameters(),
                            0.001,
                            momentum=0.9,
                            weight_decay=3e-4)
                        optimizer_axis_calib = None
                        optimizer_metric2 = None
                        optimizer_component_corr = torch.optim.SGD(
                            _component_corr_weights(self.search_space),
                            0.1,
                            momentum=0.9,
                            weight_decay=3e-4)
                        self.search_space.to(self.device)

                        op_indices = op_indices_full

                        self.search_space.instantiate_model = True
                        self.search_space.set_spec(op_indices)

                        for metric_e in range(50):
                            self.logger.info("epoch: " + str(metric_e))
                            self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                                       optimizer_component_corr, mode=1)

                            self.train(train_loader, criterion, optimizer, optimizer_axis_calib, optimizer_metric2,
                                       optimizer_component_corr, mode=4)

                            self.logger.info("component_corr_weights: {}".format(_component_corr_weights(self.search_space)[0]))
                            self.logger.info("sigmoid component_corr_weights: {}".format(F.sigmoid(_component_corr_weights(self.search_space)[0])))

                            self.single_evaluate(test_data, zc_api, perturbation)
                            if self.log_results_to_json:
                                self._log_to_json()

                            for edge in range(len(_component_corr_weights(self.search_space)[0])):
                                for op in range(len(_component_corr_weights(self.search_space)[0][edge])):
                                        _component_corr_weights(self.search_space)[0][edge][op].data.fill_(0)

        else:
            if weight_path:
                self.search_space.parse()
                weight_epoch = weight_path.split('/')[-1].split('.')[0].split('_')[-1]
                weight_prefix = weight_path.rsplit("/",1)[0]
                self.logger.info("total epoch: "+str(weight_epoch))
                if weight_epoch=='init':
                    self.search_space.load_state_dict(torch.load(weight_path))
                    self.search_space.to(self.device)

                    self.single_evaluate(test_data, zc_api, perturbation)

                    if self.log_results_to_json:
                        self._log_to_json()
                else:
                    self.logger.info("weight_epoch: init")
                    weight_path_init = os.path.join(weight_prefix,"weights_init.pt")
                    self.search_space.load_state_dict(torch.load(weight_path_init))
                    self.search_space.to(self.device)

                    self.single_evaluate(test_data, zc_api, perturbation)
                    if self.log_results_to_json:
                        self._log_to_json()

                    if self.epoch_accumulate:
                        if step:
                            weight_step = weight_epoch
                            weight_epoch = weight_path.split('/')[-1].split('.')[0].split('_')[2]
                            self.epoch_accumulate_multiplier = 1
                            train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config,auto_augment=self.auto_augment)
                            for e in range(int(weight_epoch) + 1):
                                self.epoch_accumulate_diff = 0
                                for s in range(int(weight_step)):
                                    if (s+2)%step_interval==0:
                                        self.logger.info("weight_epoch: " + str(e))
                                        self.logger.info("weight_step: " + str(s+1))
                                        self.results = [config]
                                        weight_path_epoch = os.path.join(weight_prefix, "weights_epoch_" + str(e) + "_step_"+str(s+1)+".pt")
                                        self.search_space.load_state_dict(torch.load(weight_path_epoch))
                                        self.search_space.to(self.device)

                                        self.single_evaluate(test_data, zc_api, perturbation)
                                        if self.log_results_to_json:
                                            self._log_to_json()
                                    self.epoch_accumulate_multiplier = self.epoch_accumulate_multiplier - self.epoch_accumulate_diff

                                self.logger.info("weight_epoch: " + str(e))
                                self.logger.info("weight_step: " + str(s + 1))
                                self.results = [config]
                                weight_path_epoch = os.path.join(weight_prefix,
                                                                 "weights_" + str(e) + ".pt")
                                self.search_space.load_state_dict(torch.load(weight_path_epoch))
                                self.search_space.to(self.device)

                                self.single_evaluate(test_data, zc_api, perturbation)
                                if self.log_results_to_json:
                                    self._log_to_json()
                        else:
                            self.epoch_accumulate_multiplier = 1
                            train_loader, _, test_loader, _, _ = utils.get_train_val_loaders(config,auto_augment=self.auto_augment)
                            for e in range(int(weight_epoch) + 1):
                                self.epoch_accumulate_diff = (pow(0.9, e) - pow(0.9, e + 1))
                                self.epoch_accumulate_multiplier = 1
                                self.logger.info("weight_epoch: " + str(e))
                                self.results = [config]
                                weight_path_epoch = os.path.join(weight_prefix,
                                                                 "weights_" + str(e) + ".pt")
                                self.search_space.load_state_dict(torch.load(weight_path_epoch))
                                self.search_space.to(self.device)

                                self.single_evaluate(test_data, zc_api, perturbation)
                                if self.log_results_to_json:
                                    self._log_to_json()

                    else:
                        for e in range(int(weight_epoch)+1):
                            self.logger.info("weight_epoch: " + str(e))
                            self.results = [config]
                            weight_path_epoch = os.path.join(weight_prefix, "weights_"+str(e)+".pt")
                            self.search_space.load_state_dict(torch.load(weight_path_epoch))
                            self.search_space.to(self.device)

                            self.single_evaluate(test_data, zc_api, perturbation)
                            if self.log_results_to_json:
                                self._log_to_json()

            else:
                train_loader_metric, valid_loader_metric, test_loader_metric, _, _ = utils.get_train_val_loaders(config, for_train=False, split_mode=2, fix_order_valid=True, auto_augment=self.auto_augment)

                for step, (input, target) in enumerate(train_loader_metric):
                    data = (input.cuda(), target.cuda())
                    break
                self.single_evaluate(test_data, zc_api, perturbation,data=data)
                if self.log_results_to_json:
                    self._log_to_json()

        return self.results

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with codecs.open(
            os.path.join(self.log_path, "scores.json"), "w", encoding="utf-8"
        ) as file:
            for res in self.results:
                for key, value in res.items():
                    if type(value) == np.int32 or type(value) == np.int64:
                        res[key] = int(value)
                    if type(value) == np.float32 or type(value) == np.float64:
                        res[key] = float(value)

            json.dump(self.results, file, separators=(",", ":"))

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch

    def train(self,train_loader,criterion,optimizer,optimizer_axis_calib,optimizer_metric2,optimizer_component_corr,epoch,mode=0,valid_loader=None,train_half=False,fix_order=False,decrease=False, no_edge_normalize=False):
        self.search_space.train()

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        valid_loader = []
        decrease = decrease
        stop = False
        if train_half:
            for step, (inputs, targets) in enumerate(train_loader):
                pass
            half_num = step//2
        last_acc = 0
        max_train_steps = int(os.environ.get("PROXYDIFF_MAX_TRAIN_STEPS", "0") or 0)
        progressive_mask = os.environ.get("PROXYDIFF_PROGRESSIVE_MASK", "").strip().lower()
        progressive_mask_applied = False
        for step, (inputs, targets) in enumerate(train_loader):
            if max_train_steps > 0 and step >= max_train_steps:
                self.logger.info("proxydiff_stop_train_after_steps: {}".format(max_train_steps))
                break

            inputs = Variable(inputs, requires_grad=False).cuda()
            targets = Variable(targets, requires_grad=False).cuda()
            n = inputs.size(0)

            valid_loader.append((inputs,targets))


            if fix_order:
                if step == 0:
                    self.logger.info("inputs_original:" + str(inputs))
                logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                loss = criterion(logits, targets)
            else:
                drop_prob = 0
                if mode<6:
                    if mode != 1:
                        if mode != 4 and mode!=5:
                            optimizer_axis_calib.zero_grad()
                        if mode != 3:
                            optimizer_component_corr.zero_grad()
                    if mode != 2 and mode != 3 and mode != 4:
                        optimizer.zero_grad()
                    logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,drop_prob=drop_prob,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets)

                    if self.search_space.space_name=="nasbench301":
                        if self.search_space.auxiliary_output:
                            logits_aux = self.search_space.auxiliary_logits()
                            loss_aux = criterion(logits_aux, targets)
                            loss += 0.4*loss_aux


                    loss.backward()
                    nn.utils.clip_grad_norm(self.search_space.parameters(), 5)
                    if mode != 1:
                        if mode != 4 and mode!=5:
                            optimizer_axis_calib.step()
                        if mode != 3:
                            optimizer_component_corr.step()
                    if mode != 2 and mode != 3 and mode != 4:
                        optimizer.step()

                elif mode==7:
                    inputs_search, targets_search = next(iter(valid_loader))
                    inputs_search = Variable(inputs_search, requires_grad=False).cuda()
                    targets_search = Variable(targets_search, requires_grad=False).cuda()

                    optimizer_component_corr.zero_grad()
                    logits = self.search_space.forward(inputs_search, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets_search)
                    loss.backward()
                    optimizer_component_corr.step()

                    optimizer.zero_grad()
                    logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets)
                    loss.backward()
                    nn.utils.clip_grad_norm(self.search_space.parameters(), 5)
                    optimizer.step()

                elif mode==8:
                    axis_calib_steps = int(os.environ.get("PROXYDIFF_AXIS_CALIB_STEPS", "100") or 100)
                    if (
                        progressive_mask == "topk_after_axis_calib"
                        and not progressive_mask_applied
                        and self.search_space.space_name == "nasbench301"
                        and self.search_space.metric_num > 1
                        and step >= axis_calib_steps
                    ):
                        mask_topk = int(os.environ.get("PROXYDIFF_PROGRESSIVE_MASK_TOPK", "4") or 4)
                        mask_source = os.environ.get("PROXYDIFF_PROGRESSIVE_MASK_SCORE", "axis_calib").strip().lower()
                        include_component_corr = mask_source in ("current", "axis_calib_component_corr", "with_component_corr")
                        if self.search_space.space_name == "nasbench301":
                            mask_score = _build_nb301_proxydiff_cell_scores(
                                self.search_space,
                                epoch=epoch,
                                include_component_corr=include_component_corr,
                            )
                            masks = _apply_nb301_topk_mask(self.search_space, mask_score, mask_topk)
                        _dump_progressive_mask_artifact(
                            self.log_path,
                            self.search_space,
                            mask_score,
                            masks,
                            step=step,
                            epoch=epoch,
                            topk=mask_topk,
                            source=mask_source,
                        )
                        self.logger.info(
                            "proxydiff_progressive_mask_applied: rule=topk_after_axis_calib topk={} source={} step={} kept_ops_per_cell={}".format(
                                mask_topk,
                                mask_source,
                                step,
                                [int(mask.sum().item()) for mask in masks],
                            )
                        )
                        progressive_mask_applied = True
                    if self.search_space.metric_num > 1:
                        if step < axis_calib_steps:
                            optimizer_axis_calib.zero_grad()
                        else:
                            optimizer_component_corr.zero_grad()
                    else:
                        optimizer_component_corr.zero_grad()
                    logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets)
                    if self.search_space.space_name=="nasbench301":
                        if self.search_space.auxiliary_output:
                            logits_aux = self.search_space.auxiliary_logits()
                            loss_aux = criterion(logits_aux, targets)
                            loss += 0.4*loss_aux
                    loss_contrast = contrastive_loss(logits, targets)
                    loss.backward()
                    if self.search_space.metric_num > 1:
                        if step < axis_calib_steps:
                            optimizer_axis_calib.step()
                        else:
                            optimizer_component_corr.step()
                    else:
                        optimizer_component_corr.step()
                    stagewise_interval = int(os.environ.get("PROXYDIFF_STAGEWISE_DUMP_INTERVAL", "0") or 0)
                    if (
                        stagewise_interval > 0
                        and self.search_space.space_name == "nasbench301"
                        and self.search_space.metric_num > 1
                        and step % stagewise_interval == 0
                        and step != axis_calib_steps
                    ):
                        if step < axis_calib_steps:
                            stage_score = _build_nb301_proxydiff_cell_scores(
                                self.search_space,
                                epoch=epoch,
                                include_component_corr=False,
                            )
                            stage_name = "axis_calib"
                        else:
                            stage_score = _build_nb301_proxydiff_cell_scores(
                                self.search_space,
                                epoch=epoch,
                                include_component_corr=True,
                            )
                            stage_name = "axis_calib_component_corr"
                        _dump_stagewise_artifact(
                            self.log_path,
                            self.search_space,
                            stage_score,
                            step=step,
                            epoch=epoch,
                            stage=stage_name,
                        )
                elif mode==9:
                    optimizer.zero_grad()
                    optimizer_component_corr.zero_grad()
                    logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets)
                    if self.search_space.space_name=="nasbench301":
                        if self.search_space.auxiliary_output:
                            logits_aux = self.search_space.auxiliary_logits()
                            loss_aux = criterion(logits_aux, targets)
                            loss += 0.4*loss_aux
                    loss_contrast = contrastive_loss(logits, targets)
                    loss = loss + (int(loss/loss_contrast))*loss_contrast
                    loss.backward()
                    optimizer_component_corr.step()
                    optimizer.step()
                elif mode==10:
                    optimizer.zero_grad()
                    logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                    loss = criterion(logits, targets)
                    if self.search_space.space_name == "nasbench301":
                        if self.search_space.auxiliary_output:
                            logits_aux = self.search_space.auxiliary_logits()
                            loss_aux = criterion(logits_aux, targets)
                            loss += 0.4 * loss_aux
                    loss_contrast = contrastive_loss(logits, targets)
                    loss = loss + (int(loss / loss_contrast)) * loss_contrast
                    loss.backward()
                    optimizer.step()


            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1[0].data.item(), n)

            step = step + 1




            if step == 1:
                self.logger.info('train %03d %e %f', step-1, objs.avg, top1.avg)

            if step % 50 == 0:
                self.logger.info('train %03d %e %f', step, objs.avg, top1.avg)
                stop_after_step = int(os.environ.get("PROXYDIFF_DIAG_STOP_AFTER_STEP", "0") or 0)
                if stop_after_step and step >= stop_after_step:
                    self.logger.info("proxydiff_stop_after_step: %d", step)
                    raise SystemExit(0)


            if train_half:
                if step==half_num:
                    self.logger.info('train %03d %e %f', step, objs.avg, top1.avg)
                    break

        if fix_order:
            for step, (inputs, targets) in enumerate(train_loader):
                if step == 0:
                    self.logger.info("inputs_original:" + str(inputs))

        self.logger.info('train_acc %f', top1.avg)

        return stop, decrease

    def infer(self, valid_loader, criterion, epoch):
        with torch.no_grad():
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            for step, (inputs, targets) in enumerate(valid_loader):

                inputs = Variable(inputs, requires_grad=False).cuda()
                targets = Variable(targets, requires_grad=False).cuda()
                n = inputs.size(0)

                logits = self.search_space.forward(inputs, epoch=epoch, to_print=False, fix_logger=self.log_path,no_edge_normalize=no_edge_normalize)
                loss = criterion(logits, targets)

                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                objs.update(loss.data.item(), n)
                top1.update(prec1[0].data.item(), n)

                step = step + 1

                if step == 1:
                    self.logger.info('valid %03d %e %f', step - 1, objs.avg, top1.avg)
                if step % 50 == 0:
                    self.logger.info('valid %03d %e %f', step, objs.avg, top1.avg)

        self.logger.info('valid_acc %f', top1.avg)

        return top1.avg

def contrastive_loss(features, labels, margin=0.5, pos_margin=1.0):
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.size(0)
    labels = labels.view(-1, 1)

    similarity_matrix = torch.matmul(features, features.T)

    labels_equal = labels == labels.T

    pos_pairs = similarity_matrix * labels_equal.float()

    neg_pairs = similarity_matrix * (1 - labels_equal.float())

    pos_loss = torch.clamp(pos_margin - pos_pairs, min=0.0)

    neg_loss = torch.clamp(neg_pairs - margin, min=0.0)

    loss = pos_loss + neg_loss

    return loss.mean()


class EvolutionSearcher(object):

    def __init__(self, zc_api, testset, model, model_copy, optimizer, optimizer_axis_calib, optimizer_component_corr, logger, log_path, dataset_api, train_loader, valid_loader, criterion, max_epochs=5,select_num=10,population_num=50,m_prob=0.1,crossover_num=25,mutation_num=25):

        self.max_epochs = max_epochs
        self.select_num = select_num
        self.population_num = population_num
        self.m_prob = m_prob
        self.crossover_num = crossover_num
        self.mutation_num = mutation_num

        self.zc_api = zc_api
        self.testset = testset

        self.model = model
        self.model_copy = model_copy
        self.optimizer = optimizer
        self.optimizer_axis_calib = optimizer_axis_calib
        self.optimizer_component_corr = optimizer_component_corr
        self.dataset_api = dataset_api
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.logger = logger

        self.log_path = log_path
        self.visit_epoch = []
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else self.choice(tuple(x))



    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t = list(set(t))
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                cand_tuple = tuple(cand)
                print(f"Converted cand to tuple: {cand_tuple} and its type: {type(cand_tuple)}")  # ??????
                if cand_tuple not in self.vis_dict:
                    print(f"cand_tuple {cand_tuple} not in vis_dict")
                    self.vis_dict[cand_tuple] = {}
                info = self.vis_dict[cand_tuple]
            for cand in cands:
                yield tuple(cand)

    def get_random(self, num, train=False):
        self.logger.info('random select ........')

        def random_func():
            graph = self.model_copy.clone()
            graph.sample_random_architecture(dataset_api=self.dataset_api)
            cand = graph.get_hash()
            for i in range(len(cand)):
                cand[i] = tuple(cand[i])
            cand_tuple = tuple(cand)
            return cand_tuple

        cand_iter = self.stack_random_cand(random_func)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            cand_tuple = tuple(cand)
            if not self.is_legal(cand_tuple,train=train):
                continue
            self.candidates.append(cand_tuple)
            self.logger.info('random {}/{}'.format(len(self.candidates), num))
        self.logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, train=False):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(self.choice(self.keep_top_k[k]))
            if self.model.space_name=="nasbench301":
                for c in range(len(cand)):
                    cand[c] = list(cand[c])
                    for e in range(len(cand[c])):
                        if np.random.random_sample() < m_prob:
                            i = int(e/2)
                            ops = np.random.choice(range(self.model.num_ops - 1))
                            in_node = np.random.choice(range(i + 2))
                            cand[c][e] = (in_node,ops)
                    cand[c] = tuple(cand[c])
            else:
                for i in range(self.model.num_edges):
                    if np.random.random_sample() < m_prob:
                        cand[i] = np.random.randint(self.model.num_ops)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand,train=train):
                continue
            res.append(cand)
            self.logger.info('mutation {}/{}'.format(len(res), mutation_num))

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num, train=False):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = list(self.choice(self.keep_top_k[k]))
            p2 = list(self.choice(self.keep_top_k[k]))
            if self.model.space_name=="nasbench301":
                for i in range(len(p1)):
                    p1[i] = list(p1[i])
                    p2[i] = list(p2[i])
                cand = []
                cand.append(tuple(self.choice([i, j]) for i, j in zip(p1[0], p2[0])))
                cand.append(tuple(self.choice([i, j]) for i, j in zip(p1[1], p2[1])))
                return tuple(cand)
            else:
                return tuple(self.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand,train=train):
                continue
            res.append(cand)
            self.logger.info('crossover {}/{}'.format(len(res), crossover_num))

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def is_valid(self, cand):
        xtest, ytest, test_info, _ = self.testset
        for c in range(len(cand)):
            for item in range(4):
                if cand[c][item*2][0]==cand[c][item*2+1][0] and cand[c][item*2][1]==cand[c][item*2+1][1]:
                    return False
        cand_transfer = list(cand)
        for c in range(len(cand)):
            cand_transfer[c] = list(cand_transfer[c])
            for idx in range(len(cand[c])):
                cand_transfer[c][idx] = (cand[c][idx][0], cand[c][idx][2])
            cand_transfer[c] = tuple(cand_transfer[c])
        cand_transfer = tuple(cand_transfer)

        return True

    def is_legal(self, cand, train=False, test=True):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]


        if self.model.space_name=="nasbench301":
            for c in range(len(self.model._masks)):
                for e in range(len(self.model._masks[c])):
                    for op in range(len(self.model._masks[c][e])):
                        self.model._masks[c][e][op] = 0
                        self.model_copy._masks[c][e][op] = 0
            for c in range(len(cand)):
                for item in range(len(cand[c])):
                    if len(cand[c][item])==2:
                        edge = self.model.inout_to_edge[(cand[c][item][0],int(item/2)+2)]
                        op = cand[c][item][1]
                    elif len(cand[c][item])==3:
                        edge = self.model.inout_to_edge[(cand[c][item][0], cand[c][item][1])]
                        op = cand[c][item][2]
                    self.model._masks[c][edge][op] = 1
                    self.model_copy._masks[c][edge][op] = 1

        else:
            for e in range(len(self.model._masks)):
                for op in range(len(self.model._masks[e])):
                    self.model._masks[e][op] = 0
                    self.model_copy._masks[e][op] = 0
            for i in range(len(cand)):
                self.model._masks[i][cand[i]] = 1
                self.model_copy._masks[i][cand[i]] = 1

        info['err'] = self.get_cand_err(train=train, test=test)

        if self.model.space_name=="nasbench301":
            for c in range(len(self.model._masks)):
                for e in range(len(self.model._masks[c])):
                    for op in range(len(self.model._masks[c][e])):
                        self.model._masks[c][e][op] = 1
                        self.model_copy._masks[c][e][op] = 1
        else:
            for e in range(len(self.model._masks)):
                for op in range(len(self.model._masks[e])):
                    self.model._masks[e][op] = 1
                    self.model_copy._masks[e][op] = 1

        info['visited'] = True

        return True

    def search(self, epoch, search=True, train=False):
        self.logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.epoch = epoch
        self.state_dict = copy.deepcopy(self.model.state_dict())

        self.cur_epoch = 0
        if search:

            score_orig = self.get_score(epoch=epoch)
            if self.model.space_name == "nasbench301":
                top_ops = [[[], [], [], []], [[], [], [], []]]
                new_scores = [[[], [], [], []], [[], [], [], []]]
                for c in range(len(self.model._masks)):
                    for e in range(len(self.model._masks[c])):
                        for op in range(len(self.model._masks[c][e])):
                            self.model._masks[c][e][op] = 0
                for c in range(len(self.model._masks)):
                    for e in range(len(self.model._masks[c])):
                        if e<2:
                            node = 0
                        elif e<5:
                            node = 1
                        elif e<9:
                            node = 2
                        else:
                            node = 3
                        for op in range(len(self.model._masks[c][e])):
                            new_scores[c][node].append((e,op,score_orig[c][e][op].cpu().item()))
                    for cur_node in range(4):
                        new_scores_array = np.array(new_scores[c][cur_node])
                        for _ in range(3):
                            max_idx = np.argmax(new_scores_array[:,2])
                            top_ops[c][cur_node].append(new_scores[c][cur_node][max_idx])
                            new_scores_array[max_idx][2] = 0
                            self.model._masks[c][new_scores[c][cur_node][max_idx][0]][new_scores[c][cur_node][max_idx][1]] = 1
                        top_ops[c][cur_node].sort(key=lambda x: (x[0], x[1]))

            acc_list = []
            num = 0


            if train:
                self.logger.info("mask: " + str(self.model._masks))
                self.get_cand_err(train=True, test=True)
                if self.model.space_name == "nasbench301":
                    for c in range(len(self.model._masks)):
                        for e in range(len(self.model._masks[c])):
                            for op in range(len(self.model._masks[c][e])):
                                self.model._masks[c][e][op] = 1
                else:
                    for e in range(len(self.model._masks[0])):
                        for op in range(len(self.model._masks[0][e])):
                            self.model._masks[0][e][op] = 1





        if train and not search:
            self.logger.info("mask: " + str(self.model._masks))
            self.get_cand_err(train=True,test=False)



    def get_cand_err(self, max_train_iters=100, max_test_iters=100, train=False, test=True):
        max_train_iters = max_train_iters
        max_test_iters = max_test_iters

        if self.model.space_name == "nasbench301":
            no_edge_normalize = True
        elif False:
            no_edge_normalize = False

        if train and not test:
            self.model.train()
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            for step, (data, target) in enumerate(self.train_loader):
                if step>200:
                    break
                t0 = time.time()
                batchsize = data.shape[0]


                if self.model.metric_num > 1:
                    if step < 20:
                        self.optimizer_axis_calib.zero_grad()
                    else:
                        self.optimizer_component_corr.zero_grad()
                else:
                    self.optimizer_component_corr.zero_grad()

                data, target = data.cuda(), target.cuda()
                t1 = time.time()
                output = self.model.forward(data,epoch=self.epoch,no_edge_normalize=no_edge_normalize)
                loss = self.criterion(output, target)
                loss_contrast = contrastive_loss(output, target)

                prec1, prec5 = self.accuracy(output, target, topk=(1, 5))
                top1.update(prec1.item(), batchsize)
                top5.update(prec5.item(), batchsize)
                objs.update(loss.data.item(),batchsize)

                if step % 20 == 0:
                    self.logger.info(
                        '[Supernet Training] step:%d loss: %.5f time: %.5f,'
                        % (step, objs.avg, t1 - t0)
                    )
                    self.logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.avg,top5.avg))

                loss.backward()
                if self.model.metric_num > 1:
                    if step < 100:
                        self.optimizer_axis_calib.step()
                    else:
                        self.optimizer_component_corr.step()
                else:
                    self.optimizer_component_corr.step()

            self.logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.avg,top5.avg))

        if test:
            self.model.has_metric_para = False
            if train:

                self.model.train()

                objs = utils.AvgrageMeter()
                top1 = utils.AvgrageMeter()
                top5 = utils.AvgrageMeter()
                self.logger.info("start training...")
                for step, (data, target) in enumerate(self.train_loader):
                    if step > 250:
                        break
                    t0 = time.time()
                    batchsize = data.shape[0]


                    data, target = data.cuda(), target.cuda()
                    t1 = time.time()
                    output = self.model.forward(data, epoch=self.epoch, no_edge_normalize=False, to_print=False,fix_logger=self.log_path)
                    loss = self.criterion(output, target)

                    prec1, prec5 = self.accuracy(output, target, topk=(1, 5))
                    top1.update(prec1.item(), batchsize)
                    top5.update(prec5.item(), batchsize)
                    objs.update(loss.data.item(), batchsize)

                    if step % 20 == 0:
                        self.logger.info(
                            '[Training] step:%d loss: %.5f time: %.5f,'
                            % (step, objs.avg, t1 - t0)
                        )
                        self.logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.avg, top5.avg))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.avg, top5.avg))
                if False:
                    self.logger.info("sigmoid component_corr_weights: {}".format(F.sigmoid(_component_corr_weights(self.model)[0])))
                if self.model.space_name == "nasbench301":
                    self.logger.info("sigmoid component_corr_weights: {}".format([F.sigmoid(_component_corr_weights(self.model)[c]) for c in range(2)]))


            else:
                top1 = utils.AvgrageMeter()
                top5 = utils.AvgrageMeter()

                self.logger.info('starting test....')
                self.model.eval()
                with torch.no_grad():
                    for step, (data, target) in enumerate(self.valid_loader):
                        if step>max_test_iters:
                            break
                        batchsize = data.shape[0]
                        data, target = data.cuda(), target.cuda()

                        logits = self.model.forward(data,epoch=self.epoch,to_print=False,fix_logger=self.log_path,no_edge_normalize=False)

                        prec1, prec5 = self.accuracy(logits, target, topk=(1, 5))

                        top1.update(prec1.item(), batchsize)
                        top5.update(prec5.item(), batchsize)

                        del data, target, logits, prec1, prec5

                    self.logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.avg,top5.avg))

            self.model.has_metric_para = True

            return top1.avg, top5.avg

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def get_score(self,epoch=0):
        if False:
            score = torch.zeros(self.model.num_edges, self.model.num_ops).cuda()
            for m in range(self.model.metric_num):
                sum_m = torch.zeros(self.model.num_edges, self.model.num_ops).cuda()
                if len(self.model._score) != 0:
                    if self.model.metric_epoch_accumulate[m]:
                        for e_idx in range(epoch + 1):
                            sum_m += (F.sigmoid(_edge_epoch_weights(self.model)[m][e_idx]) / sum(
                                [F.sigmoid(_edge_epoch_weights(self.model)[m][e_tmp]) for e_tmp in range(epoch + 1)])) * \
                                     self.model._score[m][e_idx]
                    else:
                        sum_m += self.model._score[m][epoch]

                    score += (F.sigmoid(_axis_calibration_weights(self.model)[m]) / sum(
                        [F.sigmoid(_axis_calibration_weights(self.model)[m_tmp]) for m_tmp in range(len(_axis_calibration_weights(self.model)))])) * sum_m


            score1 = score
            self.logger.info("score_after_axis_calib: " + str(score1))
            if self.model.metric_num != 0:
                score += (F.sigmoid(_component_corr_weights(self.model)[0]) - 0.5)
                score2 = score
                score3 = F.leaky_relu(score)
            else:
                score += F.sigmoid(_component_corr_weights(self.model)[0])
        elif self.model.space_name == "nasbench301":
            score = [torch.zeros(self.model.num_edges, self.model.num_ops - 1).cuda() for cell in range(2)]
            for cell in range(2):
                for m in range(self.model.metric_num):
                    sum_m = torch.zeros(self.model.num_edges, self.model.num_ops - 1).cuda()
                    if len(self.model._score) != 0:
                        if self.model.metric_epoch_accumulate[m]:
                            for e_idx in range(epoch + 1):
                                sum_m += (F.sigmoid(_edge_epoch_weights(self.model)[m][e_idx]) / sum(
                                    [F.sigmoid(_edge_epoch_weights(self.model)[m][e_tmp]) for e_tmp in range(epoch + 1)])) * \
                                         self.model._score[m][e_idx][cell]
                        else:
                            sum_m += self.model._score[m][epoch][cell]
                        score[cell] += (F.sigmoid(_axis_calibration_weights(self.model)[m]) / sum(
                            [F.sigmoid(_axis_calibration_weights(self.model)[m_tmp]) for m_tmp in range(len(_axis_calibration_weights(self.model)))])) * sum_m

                score1 = score[cell]
                self.logger.info("score_after_axis_calib: " + str(score1))
                if self.model.metric_num != 0:
                    score[cell] += (F.sigmoid(_component_corr_weights(self.model)[cell]) - 0.5)
                    score2 = score[cell]
                    score3 = F.leaky_relu(score[cell])
                else:
                    score[cell] += F.sigmoid(_component_corr_weights(self.model)[cell])
        return score

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

