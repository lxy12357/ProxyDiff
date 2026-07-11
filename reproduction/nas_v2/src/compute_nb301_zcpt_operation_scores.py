#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import collections.abc
import gc
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms


WORKTREE = Path(__file__).resolve().parents[3]
ENV_DEP_ROOT = os.environ.get("PROXYDIFF_NAS_DEP_ROOT")
if ENV_DEP_ROOT:
    REPRO_ROOT = Path(ENV_DEP_ROOT).expanduser().resolve()
elif (WORKTREE.parent / "upstream_zero_cost_pt").exists():
    REPRO_ROOT = WORKTREE.parent
elif (WORKTREE / "Reproduction" / "upstream_zero_cost_pt").exists():
    REPRO_ROOT = WORKTREE / "Reproduction"
else:
    REPRO_ROOT = Path("/hdd/xiaoyun/ProxyDARTS/Reproduction")

UPSTREAM = REPRO_ROOT / "upstream_zero_cost_pt"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(REPRO_ROOT))
sys.path.insert(0, str(WORKTREE))

from sota.cnn.model_search_darts_proj import DartsNetworkProj
from sota.cnn.spaces import spaces_dict


SWAP_PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]
DARTS_NODE_START = [0, 2, 5, 9]
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
MODEL_INIT_CHANNELS = 36
MODEL_LAYERS = 20


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


def make_cifar10_transforms(cutout: bool = False, cutout_length: int = 16, cutout_prob: float = 1.0):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length, cutout_prob))
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, valid_transform


def ensure_torch_six() -> None:
    if "torch._six" in sys.modules:
        return
    shim = types.ModuleType("torch._six")
    shim.container_abcs = collections.abc
    shim.string_classes = (str,)
    shim.int_classes = (int,)
    sys.modules["torch._six"] = shim


def ensure_torch_compat_apis() -> None:
    def _symeig(x, eigenvectors=False, upper=True):
        if eigenvectors:
            return torch.linalg.eigh(x, UPLO="U" if upper else "L")
        vals = torch.linalg.eigvalsh(x, UPLO="U" if upper else "L")
        return vals, None

    torch.symeig = _symeig


def ensure_simplejson_shim() -> None:
    if "simplejson" in sys.modules:
        return
    shim = types.ModuleType("simplejson")
    shim.__dict__.update(json.__dict__)
    sys.modules["simplejson"] = shim


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def configure_reproducible_backend(method: str) -> None:
    """Set backend flags for repeatable NB301 ZCPT score computation."""
    _ = method
    torch.backends.cudnn.benchmark = env_flag("PROXYDIFF_CUDNN_BENCHMARK", True)
    torch.backends.cudnn.deterministic = env_flag("PROXYDIFF_CUDNN_DETERMINISTIC", False)
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = env_flag("PROXYDIFF_CUDNN_ALLOW_TF32", True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = env_flag("PROXYDIFF_TORCH_MATMUL_TF32", True)
    if env_flag("PROXYDIFF_TORCH_DETERMINISTIC_ALGOS", False):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def safe_relative_delta(origin: float, masked: float) -> float:
    if not (np.isfinite(origin) and np.isfinite(masked)):
        return float("nan")
    if abs(float(origin)) < 1e-12:
        return float("nan")
    return float((float(origin) - float(masked)) / float(origin))


def tenas_delta_rank_items(raw_items: list[dict], ntk_key: str = "ntk_delta", rn_key: str = "rn_delta"):
    ntk_values = [float(item[ntk_key]) for item in raw_items]
    rn_values = [float(item[rn_key]) for item in raw_items]
    finite_ntk = [v for v in ntk_values if np.isfinite(v)]
    finite_rn = [v for v in rn_values if np.isfinite(v)]
    ntk_floor = min(finite_ntk) - 1.0 if finite_ntk else -1.0
    rn_ceiling = max(finite_rn) + 1.0 if finite_rn else 1.0
    ntk_for_rank = [v if np.isfinite(v) else ntk_floor for v in ntk_values]
    rn_for_rank = [v if np.isfinite(v) else rn_ceiling for v in rn_values]
    n_candidates = len(raw_items)
    ntk_prune_ranks = n_candidates - average_ranks(ntk_for_rank)
    rn_prune_ranks = average_ranks(rn_for_rank)
    return ntk_prune_ranks, rn_prune_ranks


def corr(x, y, method):
    if len(x) != len(y) or len(x) < 2:
        return None
    if method == "spearman":
        x = average_ranks(x)
        y = average_ranks(y)
    else:
        try:
            from scipy.stats import kendalltau

            v = kendalltau(x, y).correlation
            return None if math.isnan(v) else float(v)
        except Exception:
            return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return None if math.isnan(v) else v


def load_pool(path):
    data = torch.load(path, map_location="cpu")
    return list(data[0]), [float(x) for x in data[1]]


def to_genotype(arch):
    def convert_cell(cell):
        return [(SWAP_PRIMITIVES[int(op_id)], int(input_id)) for input_id, op_id in cell]

    normal, reduce = arch
    return convert_cell(normal), convert_cell(reduce)


def zcpt_cifar_loader(data_root: Path, batch_size: int, seed: int, validate_rounds: int = 10):
    train_transform, _ = make_cifar10_transforms(cutout=False, cutout_length=16, cutout_prob=1.0)
    train_data = dset.CIFAR10(root=str(data_root), train=True, download=False, transform=train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(validate_rounds * batch_size))
    return torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=0,
    )


def make_projected_model(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    criterion = nn.CrossEntropyLoss().to(device)
    args = SimpleNamespace(search_space="s5", learning_rate=0.025, momentum=0.9, weight_decay=3e-4, nesterov=False)
    return DartsNetworkProj(MODEL_INIT_CHANNELS, 10, MODEL_LAYERS, criterion, spaces_dict["s5"], args).to(device)


def apply_projected_cell(model, cell_arch, cell_type):
    with torch.no_grad():
        model.proj_weights[cell_type].zero_()
        model.candidate_flags[cell_type].fill_(False)
        model.candidate_flags_edge[cell_type].fill_(False)
        for node in range(4):
            chosen = []
            for input_id, op_id in cell_arch[node * 2 : (node + 1) * 2]:
                eid = DARTS_NODE_START[node] + int(input_id)
                chosen.append(eid)
                model.proj_weights[cell_type][eid, int(op_id)] = 1.0
            if node > 0:
                model.nid2selected_eids[cell_type][node - 1] = chosen


def apply_projected_arch(model, arch):
    model.reset_arch_parameters()
    normal, reduce = arch
    apply_projected_cell(model, list(normal), "normal")
    apply_projected_cell(model, list(reduce), "reduce")


def ablate_single_projected_op(model, cell_type: str, eid: int, opid: int):
    """Use the upstream projection mask semantics, but in a fixed neutral context."""
    with torch.no_grad():
        weights = model.get_softmax()[cell_type].detach().clone()
        weights[eid, opid] = 0.0
        model.proj_weights[cell_type][eid].copy_(weights[eid])
        model.candidate_flags[cell_type][eid] = False


def arch_component_keys(arch):
    keys = []
    for cell_type, cell in (("normal", arch[0]), ("reduce", arch[1])):
        cell = list(cell)
        for item_idx, (input_id, op_id) in enumerate(cell):
            node = item_idx // 2
            eid = DARTS_NODE_START[node] + int(input_id)
            keys.append((cell_type, int(eid), int(op_id)))
    return keys


def aggregate_arch_scores_from_ops(archs, op_score_map, missing_value: float = float("-inf")):
    scores = []
    for arch in archs:
        parts = []
        for key in arch_component_keys(arch):
            value = op_score_map.get(key)
            if value is None or not np.isfinite(value):
                parts = []
                break
            parts.append(float(value))
        scores.append(float(sum(parts)) if parts else missing_value)
    return scores


def module_l2_norm_score(module: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for param in module.parameters():
            total += float(torch.sum(param.detach() ** 2).item())
    return total


def module_param_count_score(module: nn.Module) -> float:
    return float(sum(param.numel() for param in module.parameters()))


def projected_candidate_op_l2_score(model: nn.Module, cell_type: str, eid: int, opid: int) -> float:
    """Official l2_norm over the parameters removed by ablating this candidate op.

    Runtime projection masks affect the forward path but do not delete parameters.
    For a parameter-only proxy, the official score difference for removing a
    candidate op is therefore the L2 norm of that op's own parameters across all
    cells of the requested type.
    """
    want_reduction = cell_type == "reduce"
    total = 0.0
    for cell in model.cells:
        if bool(cell.reduction) != want_reduction:
            continue
        mixed_op = cell._ops[int(eid)]
        op_module = mixed_op._ops[int(opid)]
        total += module_l2_norm_score(op_module)
    return total


def projected_candidate_op_param_score(model: nn.Module, cell_type: str, eid: int, opid: int) -> float:
    want_reduction = cell_type == "reduce"
    total = 0.0
    for cell in model.cells:
        if bool(cell.reduction) != want_reduction:
            continue
        mixed_op = cell._ops[int(eid)]
        op_module = mixed_op._ops[int(opid)]
        total += module_param_count_score(op_module)
    return total


def network_scale_label() -> str:
    return f"DARTS projected network, {MODEL_LAYERS} cells, C={MODEL_INIT_CHANNELS}"


def protocol_label(score_mode: str) -> str:
    scale = f"{MODEL_LAYERS}cell_c{MODEL_INIT_CHANNELS}"
    if score_mode == "op_ablation":
        return f"zcpt_single_op_ablation_prior_{scale}"
    return f"zcpt_projected_darts_outer_protocol_{scale}"


def write_operation_scores(out_dir: Path, method: str, seed: int, batch_size: int, baseline_score, records):
    if method == "te_nas":
        score_definition = (
            "TE-NAS component score follows official prune semantics: for each cell_type independently, compute "
            "origin and masked NTK/RN scores under the same projected supernet state, form relative deltas "
            "(origin - masked) / origin, rank higher NTK delta and lower RN delta, then use operation_score = "
            "-mean(ntk_prune_rank, rn_prune_rank); architecture score = sum selected edge-op operation scores"
        )
    elif method == "jacob":
        score_definition = (
            "Jacob component score: ablate one candidate op in the projected DARTS supernet, compute the "
            "official Zero-Cost-PT Jocab_Score summed over normal and reduce cells, and use operation_score = "
            "neutral_supernet_score - score_after_masking_one_candidate_op, matching the other scalar ZCPT proxies; "
            "architecture score = sum selected edge-op operation scores"
        )
    elif method == "nwot":
        score_definition = (
            "NWOT component score: ablate one candidate op in the projected DARTS supernet, compute the "
            "standalone NASWOT/NWOT ReLU activation-kernel logdet on the whole model, and use operation_score = "
            "neutral_supernet_score - score_after_masking_one_candidate_op; architecture score = sum selected "
            "edge-op operation scores. This row is a projected ZCPT-supernet NWOT proxy and is separate from "
            "official Zero-Cost-PT's Jocab_Score row named zcpt_jacob."
        )
    elif method == "l2_norm":
        score_definition = (
            "L2 component score: official parameter L2 norm of the candidate op module, summed across all "
            "projected DARTS cells of the requested cell_type. This is the parameter-space equivalent of "
            "neutral_supernet_score - score_after_removing_one_candidate_op; runtime forward masks are not "
            "used because they do not delete parameters."
        )
    elif method == "params":
        score_definition = (
            "Params component score: parameter count of the candidate op module, summed across all projected "
            "DARTS cells of the requested cell_type. Runtime masks are not used because masking does not delete "
            "parameters; architecture score = sum selected edge-op operation scores."
        )
    elif method == "flops":
        score_definition = (
            "FLOPs component score: ablate one candidate op in the projected DARTS supernet, compute whole-model "
            "FLOPs with zero-weight ops skipped by MixedOp.forward, and use operation_score = "
            "neutral_supernet_score - score_after_masking_one_candidate_op; architecture score = sum selected "
            "edge-op operation scores."
        )
    else:
        score_definition = (
            "Component score is stored per cell_type/candidate-op. For scalar proxies, operation_score = "
            "baseline_score - masked_score on the projected DARTS supernet; "
            "architecture score = sum selected edge-op operation scores"
        )
    payload = {
        "search_space": "nasbench301",
        "dataset": "cifar10",
        "predictor": f"official_zcpt_{method}",
        "protocol": protocol_label("op_ablation"),
        "protocol_revision": "cell_local_tenas_delta_v1_20260612" if method == "te_nas" else "cell_local_v2_20260605",
        "network_scale": network_scale_label(),
        "seed": seed,
        "batch_size": batch_size,
        "baseline_score": baseline_score,
        "score_definition": score_definition,
        "candidate_count": len(records),
        "records": records,
    }
    (out_dir / "operation_scores.json").write_text(json.dumps(payload, separators=(",", ":")))


def score_stats(values) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "n_nan": int(np.isnan(arr).sum()),
        "n_posinf": int(np.isposinf(arr).sum()),
        "n_neginf": int(np.isneginf(arr).sum()),
        "n_finite": int(np.isfinite(arr).sum()),
        "n_unique_finite": int(len(np.unique(finite))) if finite.size else 0,
        "is_constant_finite": bool(finite.size > 0 and len(np.unique(finite)) <= 1),
    }


def move_projected_state(model, device: torch.device):
    model.to(device)
    def move_value(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [move_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(move_value(v) for v in value)
        return value

    for key in list(model.proj_weights.keys()):
        model.proj_weights[key] = move_value(model.proj_weights[key])
    for key in list(model.candidate_flags.keys()):
        model.candidate_flags[key] = move_value(model.candidate_flags[key])
    for key in list(model.candidate_flags_edge.keys()):
        model.candidate_flags_edge[key] = move_value(model.candidate_flags_edge[key])
    for name in (
        "alphas_normal",
        "alphas_reduce",
        "arch_parameters",
        "_arch_parameters",
        "spaces_dict",
        "proj_weights",
        "candidate_flags",
        "candidate_flags_edge",
    ):
        if hasattr(model, name):
            setattr(model, name, move_value(getattr(model, name)))
    return model


class DeviceLoader:
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for inputs, targets in self.loader:
            yield inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.loader)


class ProjectedDartsFeatureAdapter(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def _weights(self):
        return self.base.get_projected_weights("normal"), self.base.get_projected_weights("reduce")

    def _forward_states(self, x):
        weights_normal, weights_reduce = self._weights()
        s0 = s1 = self.base.stem(x)
        cell_features = []
        for cell in self.base.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights, self.base.drop_path_prob)
            cell_features.append(s1)
        return cell_features, s1

    def forward(self, x):
        _, last = self._forward_states(x)
        out = self.base.global_pooling(last)
        logits = self.base.classifier(out.view(out.size(0), -1))
        return logits

    def forward_pre_GAP(self, x):
        _, last = self._forward_states(x)
        return last

    def extract_cell_features(self, x):
        cell_features, _ = self._forward_states(x)
        return cell_features


class ProjectedTupleNet(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x), None


def first_batch(loader, device: torch.device):
    inputs, targets = next(iter(loader))
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def first_n_batches(loader, device: torch.device, n: int):
    batches = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        batches.append(
            (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
        )
        if batch_idx + 1 >= n:
            break
    return batches


def scalar_score(value):
    if isinstance(value, (tuple, list)):
        if not value:
            return float("nan")
        value = value[0]
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value, dtype=float).mean())
    return float(value)


def patch_swap_forward_cpu_activation_concat(SWAP):
    if getattr(SWAP, "_proxydiff_cpu_activation_concat_patched", False):
        return

    original_register_hook = SWAP.register_hook

    def register_hook_with_handles(self, model):
        if getattr(self, "_proxydiff_hook_handles", None):
            return
        self._proxydiff_hook_handles = []
        for _, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self._proxydiff_hook_handles.append(module.register_forward_hook(hook=self.hook_in_forward))

    def remove_hooks(self):
        for handle in getattr(self, "_proxydiff_hook_handles", []) or []:
            handle.remove()
        self._proxydiff_hook_handles = []

    def hook_in_forward_cpu(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())

    def is_cuda_oom(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "cuda" in msg and ("out of memory" in msg or "cublas" in msg)

    def forward_cpu_activation_concat(self):
        self.interFeature = []
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            self.model.forward(self.inputs.to(model_device))
            if len(self.interFeature) == 0:
                return None
            gpu_activations = None
            cpu_activations = None
            try:
                gpu_activations = torch.cat(
                    [f.view(self.inputs.size(0), -1) for f in self.interFeature],
                    1,
                )
                self.swap.device = model_device
                self.swap.collect_activations(gpu_activations)
                return self.swap.calSWAP(self.regular_factor)
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                if gpu_activations is not None:
                    del gpu_activations
                self.swap.activations = None
                torch.cuda.empty_cache()
                cpu_activations = torch.cat(
                    [f.cpu().view(self.inputs.size(0), -1) for f in self.interFeature],
                    1,
                )
                self.swap.device = torch.device("cpu")
                self.swap.collect_activations(cpu_activations)
                return self.swap.calSWAP(self.regular_factor)
            finally:
                if gpu_activations is not None:
                    del gpu_activations
                if cpu_activations is not None:
                    del cpu_activations
                del self.interFeature
                self.interFeature = []

    SWAP._proxydiff_original_register_hook = original_register_hook
    SWAP.register_hook = register_hook_with_handles
    SWAP.remove_hooks = remove_hooks
    SWAP.hook_in_forward = hook_in_forward_cpu
    SWAP.forward = forward_cpu_activation_concat
    SWAP._proxydiff_cpu_activation_concat_patched = True


def compute_nwot_score(net: nn.Module, inputs: torch.Tensor) -> float:
    # NASWOT/NWOT activation-kernel logdet, matching NASLib/ZeroCostNAS nwot.py.
    batch_size = int(inputs.size(0))
    net.K = np.zeros((batch_size, batch_size))
    handles = []

    def counting_forward_hook(module, inp, out):
        x_in = inp[0].view(inp[0].size(0), -1)
        x_bin = (x_in > 0).float()
        k_pos = x_bin @ x_bin.t()
        k_neg = (1.0 - x_bin) @ (1.0 - x_bin.t())
        net.K = net.K + k_pos.detach().cpu().numpy() + k_neg.detach().cpu().numpy()

    for module in net.modules():
        module_type = str(type(module))
        if "ReLU" in module_type and "naslib" not in module_type:
            handles.append(module.register_forward_hook(counting_forward_hook))

    with torch.no_grad():
        net(torch.clone(inputs))
    for handle in handles:
        handle.remove()
    _, logdet = np.linalg.slogdet(net.K)
    return float(logdet)


def compute_projected_meco_score(net: nn.Module, inputs: torch.Tensor) -> float:
    # MeCo's official hook assumes every hooked module returns a tensor-like
    # feature. The projected DARTS model has a few bookkeeping modules whose
    # hook outputs are not tensors, so keep the official corr/eigenvalue core
    # and ignore non-feature outputs.
    result_list = []
    handles = []

    def corrcoef_compat(fea: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "corrcoef"):
            return torch.corrcoef(fea)
        fea = fea.float()
        fea = fea - fea.mean(dim=1, keepdim=True)
        denom_n = max(int(fea.shape[1]) - 1, 1)
        cov = torch.mm(fea, fea.t()) / float(denom_n)
        std = torch.sqrt(torch.diag(cov).clamp_min(0))
        denom = torch.outer(std, std)
        corr = cov / denom
        corr[denom == 0] = float("nan")
        return corr

    def eig_real_compat(matrix: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "eig"):
            return torch.real(torch.linalg.eig(matrix)[0])
        return torch.eig(matrix, eigenvectors=False).eigenvalues[:, 0]

    def forward_hook(module, data_input, data_output):
        out = data_output
        if isinstance(out, (tuple, list)):
            out = out[0] if out else None
        if not isinstance(out, torch.Tensor) or out.ndim < 2:
            return
        try:
            fea = out[0].detach()
            if fea.ndim == 0:
                return
            fea = fea.reshape(fea.shape[0], -1)
            if fea.shape[0] < 2:
                return
            corr = corrcoef_compat(fea)
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = eig_real_compat(corr)
            result_list.append(torch.min(values))
        except Exception:
            return

    for _, module in net.named_modules():
        handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        net(inputs)
    for handle in handles:
        handle.remove()
    if not result_list:
        return float("nan")
    results = torch.stack(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    if results.numel() == 0:
        return float("nan")
    return float(torch.sum(results).item())


def score_projected_measure_family(method: str, archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    ensure_simplejson_shim()
    sys.path.insert(0, str(REPRO_ROOT / "MeCo"))
    sys.path.insert(0, str(REPRO_ROOT / "MeCo" / "zero-cost-nas"))
    from foresight.pruners import predictive
    from foresight.pruners.measures.l2_norm import get_l2_norm_array

    device = torch.device("cuda")
    train_queue = zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed)
    inputs, targets = first_batch(train_queue, device)
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.train()
        if method == "l2_norm":
            arrays = get_l2_norm_array(model, inputs, targets, mode="param")
            score = sum(torch.sum(x) for x in arrays).item()
        elif method == "meco":
            score = compute_projected_meco_score(model, inputs)
        else:
            model.requires_feature = False
            measures = predictive.find_measures(
                model,
                train_queue,
                ("random", 1, 10),
                device,
                measure_names=[method],
            )
            score = measures[method]
        ypred.append(scalar_score(score))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_{method}: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def score_zcpt_nwot(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    device = torch.device("cuda")
    inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.train()
        ypred.append(compute_nwot_score(model, inputs))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_nwot: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def score_zcpt_jacob(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    from sota.cnn.init_projection import Jocab_Score

    device = torch.device("cuda")
    inputs, targets = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.eval()
        score = Jocab_Score(model, "normal", inputs, targets, weights=model.get_all_projected_weights("normal"))
        score += Jocab_Score(model, "reduce", inputs, targets, weights=model.get_all_projected_weights("reduce"))
        ypred.append(float(score))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_jacob: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def score_zcpt_swap(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    sys.path.insert(0, str(REPRO_ROOT / "SWAP"))
    from src.metrics.swap import SWAP
    patch_swap_forward_cpu_activation_concat(SWAP)

    def safe_network_weight_gaussian_init(module: nn.Module):
        with torch.no_grad():
            if isinstance(module, nn.Conv2d):
                if module.weight is not None:
                    nn.init.normal_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.normal_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        return module

    device = torch.device("cuda")
    inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        swap = SWAP(model=model, inputs=inputs, device=device, seed=seed)
        score_list = []
        try:
            for _ in range(32):
                model = model.apply(safe_network_weight_gaussian_init)
                swap.reinit()
                score_list.append(float(swap.forward()))
                swap.clear()
            ypred.append(float(np.mean(score_list)))
            used_archs.append(str(arch))
        finally:
            if hasattr(swap, "remove_hooks"):
                swap.remove_hooks()
        del model, swap, score_list
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_swap: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def near_reset_weights(module: nn.Module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def near_effective_rank(matrix: torch.Tensor):
    import scipy
    import scipy.stats

    if hasattr(torch.linalg, "svdvals"):
        singular_values = torch.linalg.svdvals(matrix)
    else:
        singular_values = torch.svd(matrix).S
    singular_values = singular_values / torch.sum(singular_values)
    return np.nan_to_num(math.e ** scipy.stats.entropy(singular_values.detach()))


def near_score_once_with_hook_cleanup(model: nn.Module, dataloader, layer_index=None):
    hooks = []
    activations = []
    finished = []
    called = set()

    def get_activation(layer_id):
        def hook(module, input, output):
            called.add(layer_id)
            if isinstance(output, tuple):
                output = output[0]
            size = output.shape[-1]
            if output.dim() > 2:
                output = torch.transpose(output, 1, 3).flatten(0, 2)
            activations[layer_id] = torch.cat((activations[layer_id], output), dim=0)
            if activations[layer_id].shape[0] >= activations[layer_id].shape[1]:
                start = (
                    np.random.randint(0, activations[layer_id].shape[0] // size - 1) * size
                    if activations[layer_id].shape[0] // size - 1 > 0
                    else 0
                )
                end = start + activations[layer_id].shape[1]
                activations[layer_id] = activations[layer_id][start:end]
                hooks[layer_id].remove()
                finished.append(layer_id)

        return hook

    activation_names = getattr(
        torch.nn.modules.activation,
        "__all__",
        (
            "ELU",
            "Hardshrink",
            "Hardtanh",
            "LeakyReLU",
            "LogSigmoid",
            "MultiheadAttention",
            "PReLU",
            "ReLU",
            "ReLU6",
            "RReLU",
            "SELU",
            "CELU",
            "GELU",
            "Sigmoid",
            "SiLU",
            "Mish",
            "Softplus",
            "Softshrink",
            "Softsign",
            "Tanh",
            "Tanhshrink",
            "Threshold",
        ),
    )
    activation_functions = tuple(
        getattr(torch.nn, name) for name in activation_names if hasattr(torch.nn, name)
    )
    layer_stack = [
        module
        for _, module in model.named_modules()
        if hasattr(module, "weight") or isinstance(module, activation_functions)
    ]
    if layer_index is not None:
        layer_stack = [layer_stack[layer_index]]
    try:
        for layer_id, layer in enumerate(layer_stack):
            activations.append(torch.tensor([]))
            hooks.append(layer.register_forward_hook(get_activation(layer_id)))

        for inputs, _ in dataloader:
            model(inputs)
            if len(finished) == len(called):
                break

        score = 0.0
        for activation in activations:
            if len(activation) == 0:
                continue
            score += near_effective_rank(activation)
        return score
    finally:
        for handle in hooks:
            handle.remove()


def get_near_score_with_hook_cleanup(model: nn.Module, dataloader, layer_index=None, repetitions: int = 1):
    scores = []
    for _ in range(repetitions):
        scores.append(near_score_once_with_hook_cleanup(model, dataloader, layer_index=layer_index))
        model.apply(near_reset_weights)
    return np.mean(scores)


def score_zcpt_near(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int, repetitions: int):
    sys.path.insert(0, str(REPRO_ROOT / "NEAR" / "src"))

    loader = zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed)
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, torch.device("cuda"))
        apply_projected_arch(model, arch)
        model = move_projected_state(model, torch.device("cpu"))
        model.train()
        score = float(get_near_score_with_hook_cleanup(model, loader, repetitions=repetitions, layer_index=None))
        ypred.append(score)
        used_archs.append(str(arch))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_near: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def score_zcpt_zen(archs, accs, seed: int, limit: int | None, batch_size: int):
    ensure_torch_six()
    ensure_torch_compat_apis()
    sys.path.insert(0, str(REPRO_ROOT / "ZenNAS"))
    from ZeroShotProxy.compute_zen_score import compute_nas_score

    device = torch.device("cuda")
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        adapter = ProjectedDartsFeatureAdapter(model).to(device).train()
        info = compute_nas_score(
            gpu=0,
            model=adapter,
            mixup_gamma=1e-2,
            resolution=32,
            batch_size=batch_size,
            repeat=32,
            fp16=False,
        )
        ypred.append(float(info["avg_nas_score"]))
        used_archs.append(str(arch))
        del adapter
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_zen: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def kaiming_normal_fanin_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight.data)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight.data)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


def compute_zcpt_ntk_score(model: nn.Module, resolution: int, batch_size: int, device: torch.device) -> float:
    model.train()
    grads = []
    inputs = torch.randn((batch_size, 3, resolution, resolution), device=device)
    model.zero_grad()
    logits = model(inputs)
    for sample_idx in range(batch_size):
        logits[sample_idx : sample_idx + 1].backward(torch.ones_like(logits[sample_idx : sample_idx + 1]), retain_graph=True)
        grad_parts = []
        for name, weight in model.named_parameters():
            if "weight" in name and weight.grad is not None:
                grad_parts.append(weight.grad.view(-1).detach())
        grads.append(torch.cat(grad_parts, -1))
        model.zero_grad()
    grads = torch.stack(grads, 0)
    ntk = torch.einsum("nc,mc->nm", [grads, grads])
    eigenvalues = torch.linalg.eigvalsh(ntk)
    cond = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True)
    return -1.0 * float(cond)


@torch.no_grad()
def compute_zcpt_num_lr(model: nn.Module, resolution: int, batch_size: int, device: torch.device) -> float:
    activations = []

    def hook_fn(_module, _inputs, output):
        activations.append(output.detach())

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = model(torch.randn((batch_size, 3, resolution, resolution), device=device))

    for hook in hooks:
        hook.remove()

    if not activations:
        return float("-inf")
    feature_data = torch.cat([feat.view(batch_size, -1) for feat in activations], dim=1)
    signed = torch.sign(feature_data)
    res = torch.matmul(signed.half(), (1 - signed).T.half())
    res = res + res.T
    res = 1 - torch.sign(res)
    res = res.sum(1)
    res = 1.0 / res.float()
    return float(res.sum().item())


def compute_te_nas_ntk_condition(model: nn.Module, batches) -> float:
    model.train()
    grads = []
    for inputs, _targets in batches[:1]:
        model.zero_grad()
        logits = model(inputs)
        for sample_idx in range(inputs.size(0)):
            logits[sample_idx : sample_idx + 1].backward(
                torch.ones_like(logits[sample_idx : sample_idx + 1]),
                retain_graph=True,
            )
            grad_parts = []
            for name, weight in model.named_parameters():
                if "weight" in name and weight.grad is not None:
                    grad_parts.append(weight.grad.view(-1).detach())
            if grad_parts:
                grads.append(torch.cat(grad_parts, -1))
            model.zero_grad()
    if not grads:
        return float("inf")
    grads = torch.stack(grads, 0)
    ntk = torch.einsum("nc,mc->nm", [grads, grads]).float()
    eigenvalues = torch.linalg.eigvalsh(ntk)
    cond = np.nan_to_num(
        (eigenvalues[-1] / eigenvalues[0]).item(),
        copy=True,
        nan=100000.0,
        posinf=100000.0,
        neginf=100000.0,
    )
    return float(cond)


def compute_te_nas_linear_regions(model: nn.Module, batches) -> float:
    activations = []

    def hook_fn(_module, _inputs, output):
        activations.append(output.detach())

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    total_score = 0.0
    with torch.no_grad():
        for inputs, _targets in batches[:3]:
            activations.clear()
            _ = model(inputs)
            if not activations:
                continue
            feature_data = torch.cat([feat.view(inputs.size(0), -1) for feat in activations], dim=1)
            signed = torch.sign(feature_data)
            res = torch.matmul(signed.half(), (1 - signed).T.half())
            res = res + res.T
            res = 1 - torch.sign(res)
            res = res.sum(1)
            res = 1.0 / res.float()
            total_score += float(res.sum().item())

    for hook in hooks:
        hook.remove()
    return total_score


def reset_tenas_rn_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def score_zcpt_te_nas(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    device = torch.device("cuda")
    te_batches = first_n_batches(
        zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed, validate_rounds=3),
        device,
        1,
    )
    n = len(archs) if limit is None else min(limit, len(archs))
    ntk_scores = []
    rn_scores = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        ntk_repeat_scores = []
        rn_repeat_scores = []
        for repeat_idx in range(3):
            ntk_model = make_projected_model(seed, device)
            apply_projected_arch(ntk_model, arch)
            ntk_model.apply(kaiming_normal_fanout_init)
            ntk_repeat_scores.append(compute_te_nas_ntk_condition(ntk_model, te_batches))
            del ntk_model
            torch.cuda.empty_cache()

            rn_model = make_projected_model(seed, device)
            apply_projected_arch(rn_model, arch)
            rn_model.apply(kaiming_normal_fanin_init)
            rn_model.train()
            reset_tenas_rn_seed(seed + 1701 + repeat_idx)
            rn_repeat_scores.append(compute_zcpt_num_lr(rn_model, 32, batch_size, device))
            del rn_model
            torch.cuda.empty_cache()
        ntk_scores.append(float(np.mean(ntk_repeat_scores)))
        rn_scores.append(float(np.mean(rn_repeat_scores)))
        used_archs.append(str(arch))
        if (i + 1) % 25 == 0:
            print(f"zcpt_te_nas: scored {i + 1}/{n}", flush=True)
    ypred = (average_ranks([-float(x) for x in ntk_scores]) + average_ranks(rn_scores)).tolist()
    return used_archs, accs[:n], ypred


def score_zcpt_num_lr(archs, accs, seed: int, limit: int | None, batch_size: int):
    device = torch.device("cuda")
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.apply(kaiming_normal_fanin_init)
        model.train()
        ypred.append(compute_zcpt_num_lr(model, 32, batch_size, device))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_num_lr: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def compute_flops(model: nn.Module, device: torch.device) -> float:
    dummy = torch.randn(1, 3, 32, 32, device=device)
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops / 1e6)
    except Exception:
        pass
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model, dummy).total()
    return float(flops / 1e6)


def score_zcpt_structural(method: str, archs, accs, seed: int, limit: int | None):
    device = torch.device("cuda")
    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.eval()
        if method == "flops":
            score = compute_flops(model, device)
        else:
            score = float(sum(p.numel() for p in model.parameters()) / 1e6)
        ypred.append(float(score))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_{method}: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def load_official_epe_functions(search_py: Path):
    if not search_py.exists():
        raise FileNotFoundError(f"VascoLopes/EPE-NAS search.py not found: {search_py}")
    tree = ast.parse(search_py.read_text())
    wanted = {"get_batch_jacobian", "eval_score_perclass"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    found = {node.name for node in nodes}
    if found != wanted:
        raise RuntimeError(f"Missing EPE-NAS official functions in {search_py}: {sorted(wanted - found)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "torch": torch}
    exec(compile(module, str(search_py), "exec"), namespace)
    return {name: namespace[name] for name in wanted}


def score_zcpt_epe_nas(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    device = torch.device("cuda")
    inputs, targets = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
    n = len(archs) if limit is None else min(limit, len(archs))
    official_funcs = load_official_epe_functions(REPRO_ROOT / "EPE-NAS" / "search.py")
    get_batch_jacobian = official_funcs["get_batch_jacobian"]
    eval_score_perclass = official_funcs["eval_score_perclass"]

    class OfficialEpeOutputAdapter(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor):
            return None, self.model(x)

    def compute_score(model: nn.Module) -> float:
        _, logits = model(inputs)
        n_classes = logits.shape[-1]
        jacobs, labels = get_batch_jacobian(model, inputs.detach().clone(), targets, None, None)
        jacobs_np = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)
        labels_np = labels.cpu().numpy()
        return float(eval_score_perclass(jacobs_np, [labels_np], n_classes))

    raw_scores = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        projected_model = make_projected_model(seed, device)
        apply_projected_arch(projected_model, arch)
        model = OfficialEpeOutputAdapter(projected_model)
        model.train()
        raw_scores.append(compute_score(model))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_epe_nas: scored {i + 1}/{n}", flush=True)

    arr = np.asarray(raw_scores, dtype=float)
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        finite = arr[np.isfinite(arr)]
        floor = float(np.min(finite) - max(float(np.ptp(finite)), 1.0) * 1e-6) if finite.size else -1e9
        arr[invalid_mask] = floor
        print(f"zcpt_epe_nas: replaced {invalid_count} invalid scores with bottom score {floor}", flush=True)
    return used_archs, accs[:n], arr.astype(float).tolist(), invalid_count


def score_zcpt_epsinas(
    archs,
    accs,
    seed: int,
    data_root: Path,
    limit: int | None,
    batch_size: int,
    weights: tuple[float, float],
):
    device = torch.device("cuda")
    inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
    n = len(archs) if limit is None else min(limit, len(archs))

    def prepare_seed():
        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)
        torch.cuda.manual_seed(21)

    def initialize_weights(model: nn.Module, constant: float):
        for param in model.parameters():
            nn.init.constant_(param.data, constant)

    def model_forward(model: nn.Module, x: torch.Tensor) -> np.ndarray:
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().cpu().numpy()

    def normalize_preds(preds: np.ndarray) -> np.ndarray:
        preds = preds - np.nanmin(preds)
        denom = np.nanmax(preds)
        if np.isfinite(denom) and denom != 0:
            preds = preds / denom
        preds[preds == 0] = np.nan
        return preds

    def compute_score(model: nn.Module) -> float:
        values = []
        for constant in weights:
            prepare_seed()
            initialize_weights(model, constant)
            preds = model_forward(model, inputs)
            values.append(normalize_preds(preds))
        num = np.nanmean(np.abs(values[0] - values[1]))
        den = np.nanmean(values)
        return float(num / den)

    raw_scores = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        model.train()
        raw_scores.append(compute_score(model))
        used_archs.append(str(arch))
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_epsinas: scored {i + 1}/{n}", flush=True)

    arr = np.asarray(raw_scores, dtype=float)
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        finite = arr[np.isfinite(arr)]
        floor = float(np.min(finite) - max(float(np.ptp(finite)), 1.0) * 1e-6) if finite.size else -1e9
        arr[invalid_mask] = floor
        print(f"zcpt_epsinas: replaced {invalid_count} invalid scores with bottom score {floor}", flush=True)
    return used_archs, accs[:n], arr.astype(float).tolist(), invalid_count


def eznas_initialize_module(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def eznas_desired_pattern_enforcer(filtered_modules):
    desired_pattern = ["Conv2d", "BatchNorm2d", "ReLU"]
    pattern_index = 0
    corrected = []
    for layer in filtered_modules:
        layer_type = layer.__class__.__name__
        if layer_type == desired_pattern[pattern_index]:
            corrected.append(layer)
            pattern_index = (pattern_index + 1) % len(desired_pattern)
        elif layer_type == "Conv2d":
            pattern_index = 1
    return corrected


def eznas_align_conv_bn_relu_to_official_triples(filtered_modules):
    """Convert Conv-BN-ReLU triples to official ReLU-Conv-BN indexing.

    Official EZNAS later reads `flat_layers[3*i+1].weight` as the layer weight,
    so the sequence must be ReLU, Conv2d, BatchNorm2d repeated.  The dynamic
    fallback naturally extracts Conv2d, BatchNorm2d, ReLU triples; dropping the
    first Conv/BN and final ReLU restores the same indexing convention as the
    official static `flat_layer_module_list[2:]` path.
    """
    if len(filtered_modules) >= 3 and [m.__class__.__name__ for m in filtered_modules[:3]] == ["Conv2d", "BatchNorm2d", "ReLU"]:
        aligned = filtered_modules[2:-1]
        if len(aligned) % 3 == 0 and aligned:
            return aligned
    return filtered_modules


def create_eznas_layer_sequence(model: nn.Module, sample_tensor: torch.Tensor):
    modules = [m for m in model.modules() if m.__class__.__name__ in {"Conv2d", "BatchNorm2d", "ReLU"}]
    cleaned = []
    idx = 0
    while idx < len(modules):
        if idx + 1 < len(modules) and modules[idx].__class__.__name__ == "Conv2d" and modules[idx + 1].__class__.__name__ == "Conv2d":
            cleaned.append(modules[idx + 1])
            idx += 2
        elif idx + 1 < len(modules) and modules[idx].__class__.__name__ == "Conv2d" and modules[idx + 1].__class__.__name__ != "BatchNorm2d":
            idx += 1
        else:
            cleaned.append(modules[idx])
            idx += 1
    cleaned = cleaned[2:]
    if len(cleaned) % 3 == 0 and cleaned:
        return cleaned

    execution_order = []
    hooks = []

    def forward_hook(module, _input, _output):
        execution_order.append(module)

    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))
    with torch.no_grad():
        _ = model(sample_tensor)
    for hook in hooks:
        hook.remove()
    filtered = [m for m in execution_order if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU))]
    filtered = eznas_desired_pattern_enforcer(filtered)
    filtered = eznas_desired_pattern_enforcer(filtered)
    filtered = eznas_desired_pattern_enforcer(filtered)
    return eznas_align_conv_bn_relu_to_official_triples(filtered)


class EznasHook:
    def __init__(self, module, backward=False):
        if backward:
            # Match official EZNAS.  The full backward hook path can fail on
            # A non-inplace ReLU avoids view/in-place interactions that can
            # produce degenerate activation statistics in this scorer.
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, _module, inputs, output):
        self.input = inputs
        self.output = output

    def remove(self):
        self.hook.remove()


def run_eznas_collect(model, sample_tensor, flat_layers):
    state = {}
    num_conv_layers = len(flat_layers) // 3
    for layer_idx in range(num_conv_layers):
        state[f"layer{layer_idx}"] = {}

    def collect(process_noise=False, perturb=0.0):
        hook_f = [EznasHook(layer) for layer in flat_layers]
        hook_b = [EznasHook(layer, backward=True) for layer in flat_layers]
        model.zero_grad(set_to_none=True)
        data_sample = sample_tensor.detach().clone()
        if process_noise:
            data_sample = torch.randn_like(data_sample)
        if perturb:
            data_sample = data_sample + perturb ** 0.5 * torch.randn_like(data_sample)
        data_sample.requires_grad_(True)
        model_out = model(data_sample)
        out = model_out[0] if isinstance(model_out, tuple) else model_out
        out.backward(torch.ones_like(out), retain_graph=True)
        for layer_idx in range(num_conv_layers):
            key = f"layer{layer_idx}"
            try:
                if not process_noise and perturb == 0.0:
                    state[key]["inpactdata"] = hook_f[3 * layer_idx].input[0].clone().detach()
                    state[key]["inpactgraddata"] = hook_b[3 * layer_idx].input[0].clone().detach()
                    state[key]["wt"] = flat_layers[3 * layer_idx + 1].weight.clone().detach()
                    state[key]["wtgraddata"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach()
                    state[key]["preactdata"] = hook_f[3 * layer_idx + 1].input[0].clone().detach()
                    state[key]["preactgraddata"] = hook_b[3 * layer_idx + 1].input[0].clone().detach()
                    state[key]["actdata"] = hook_f[3 * layer_idx + 2].output.clone().detach()
                    state[key]["actgraddata"] = hook_b[3 * layer_idx + 2].output[0].clone().detach()
                elif process_noise and perturb == 0.0:
                    state[key]["inpactnoise"] = hook_f[3 * layer_idx].input[0].clone().detach()
                    state[key]["inpactgradnoise"] = hook_b[3 * layer_idx].input[0].clone().detach()
                    state[key]["wtgradnoise"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach()
                    state[key]["preactnoise"] = hook_f[3 * layer_idx + 1].input[0].clone().detach()
                    state[key]["preactgradnoise"] = hook_b[3 * layer_idx + 1].input[0].clone().detach()
                    state[key]["actnoise"] = hook_f[3 * layer_idx + 2].output.clone().detach()
                    state[key]["actgradnoise"] = hook_b[3 * layer_idx + 2].output[0].clone().detach()
                elif not process_noise and perturb:
                    state[key]["inpactperturb"] = hook_f[3 * layer_idx].input[0].clone().detach() - state[key]["inpactdata"]
                    state[key]["inpactgradperturb"] = hook_b[3 * layer_idx].input[0].clone().detach() - state[key]["inpactgraddata"]
                    state[key]["wtgradperturb"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach() - state[key]["wtgraddata"]
                    state[key]["preactperturb"] = hook_f[3 * layer_idx + 1].input[0].clone().detach() - state[key]["preactdata"]
                    state[key]["preactgradperturb"] = hook_b[3 * layer_idx + 1].input[0].clone().detach() - state[key]["preactgraddata"]
                    state[key]["actperturb"] = hook_f[3 * layer_idx + 2].output.clone().detach() - state[key]["actdata"]
                    state[key]["actgradperturb"] = hook_b[3 * layer_idx + 2].output[0].clone().detach() - state[key]["actgraddata"]
            except Exception:
                state[key] = {}
        for hook in hook_f + hook_b:
            hook.remove()
        model.zero_grad()

    collect(process_noise=False, perturb=0.0)
    collect(process_noise=True, perturb=0.0)
    collect(process_noise=False, perturb=0.01)
    return state


def eznas_darts_layer_score(layer_state):
    if not layer_state or "wtgraddata" not in layer_state:
        return 0.0
    # Keep the official EZNAS expression, but run logdet on CPU to avoid
    a = torch.sign(layer_state["wtgraddata"]).detach().cpu()
    try:
        a = torch.logdet(a)
        a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
        a = torch.sigmoid(a)
        a = torch.norm(a.float(), p="fro")
        return float(torch.sum(a).item() / max(1, a.numel()))
    except Exception:
        return None


def score_zcpt_eznas(archs, accs, seed: int, data_root: Path, limit: int | None, batch_size: int):
    device = torch.device("cuda")
    sample = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)[0]

    def initialize_module(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def desired_pattern_enforcer(filtered_modules):
        desired_pattern = ["Conv2d", "BatchNorm2d", "ReLU"]
        pattern_index = 0
        corrected = []
        for layer in filtered_modules:
            layer_type = layer.__class__.__name__
            if layer_type == desired_pattern[pattern_index]:
                corrected.append(layer)
                pattern_index = (pattern_index + 1) % len(desired_pattern)
            elif layer_type == "Conv2d":
                pattern_index = 1
        return corrected

    def create_layer_sequence(model: nn.Module, sample_tensor: torch.Tensor):
        modules = [m for m in model.modules() if m.__class__.__name__ in {"Conv2d", "BatchNorm2d", "ReLU"}]
        cleaned = []
        idx = 0
        while idx < len(modules):
            if idx + 1 < len(modules) and modules[idx].__class__.__name__ == "Conv2d" and modules[idx + 1].__class__.__name__ == "Conv2d":
                cleaned.append(modules[idx + 1])
                idx += 2
            elif idx + 1 < len(modules) and modules[idx].__class__.__name__ == "Conv2d" and modules[idx + 1].__class__.__name__ != "BatchNorm2d":
                idx += 1
            else:
                cleaned.append(modules[idx])
                idx += 1
        cleaned = cleaned[2:]
        if len(cleaned) % 3 == 0 and cleaned:
            return cleaned

        execution_order = []
        hooks = []

        def forward_hook(module, _input, _output):
            execution_order.append(module)

        for module in model.modules():
            hooks.append(module.register_forward_hook(forward_hook))
        with torch.no_grad():
            _ = model(sample_tensor)
        for hook in hooks:
            hook.remove()
        filtered = [m for m in execution_order if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU))]
        filtered = desired_pattern_enforcer(filtered)
        filtered = desired_pattern_enforcer(filtered)
        filtered = desired_pattern_enforcer(filtered)
        return eznas_align_conv_bn_relu_to_official_triples(filtered)

    class Hook:
        def __init__(self, module, backward=False):
            if backward:
                # Match official EZNAS.  See EznasHook above.
                self.hook = module.register_backward_hook(self.hook_fn)
            else:
                self.hook = module.register_forward_hook(self.hook_fn)
            self.input = None
            self.output = None

        def hook_fn(self, _module, inputs, output):
            self.input = inputs
            self.output = output

        def remove(self):
            self.hook.remove()

    def run_and_collect(model, sample_tensor, flat_layers):
        state = {}
        num_conv_layers = len(flat_layers) // 3
        for layer_idx in range(num_conv_layers):
            state[f"layer{layer_idx}"] = {}

        def collect(process_noise=False, perturb=0.0):
            hook_f = [Hook(layer) for layer in flat_layers]
            hook_b = [Hook(layer, backward=True) for layer in flat_layers]
            data_sample = sample_tensor.detach().clone()
            if process_noise:
                data_sample = torch.randn_like(data_sample)
            if perturb:
                data_sample = data_sample + perturb ** 0.5 * torch.randn_like(data_sample)
            data_sample.requires_grad_(True)
            model_out = model(data_sample)
            out = model_out[0] if isinstance(model_out, tuple) else model_out
            out.backward(torch.ones_like(out))
            for layer_idx in range(num_conv_layers):
                key = f"layer{layer_idx}"
                try:
                    if not process_noise and perturb == 0.0:
                        state[key]["inpactdata"] = hook_f[3 * layer_idx].input[0].clone().detach()
                        state[key]["inpactgraddata"] = hook_b[3 * layer_idx].input[0].clone().detach()
                        state[key]["wt"] = flat_layers[3 * layer_idx + 1].weight.clone().detach()
                        state[key]["wtgraddata"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach()
                        state[key]["preactdata"] = hook_f[3 * layer_idx + 1].input[0].clone().detach()
                        state[key]["preactgraddata"] = hook_b[3 * layer_idx + 1].input[0].clone().detach()
                        state[key]["actdata"] = hook_f[3 * layer_idx + 2].output.clone().detach()
                        state[key]["actgraddata"] = hook_b[3 * layer_idx + 2].output[0].clone().detach()
                    elif process_noise and perturb == 0.0:
                        state[key]["inpactnoise"] = hook_f[3 * layer_idx].input[0].clone().detach()
                        state[key]["inpactgradnoise"] = hook_b[3 * layer_idx].input[0].clone().detach()
                        state[key]["wtgradnoise"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach()
                        state[key]["preactnoise"] = hook_f[3 * layer_idx + 1].input[0].clone().detach()
                        state[key]["preactgradnoise"] = hook_b[3 * layer_idx + 1].input[0].clone().detach()
                        state[key]["actnoise"] = hook_f[3 * layer_idx + 2].output.clone().detach()
                        state[key]["actgradnoise"] = hook_b[3 * layer_idx + 2].output[0].clone().detach()
                    elif not process_noise and perturb:
                        state[key]["inpactperturb"] = hook_f[3 * layer_idx].input[0].clone().detach() - state[key]["inpactdata"]
                        state[key]["inpactgradperturb"] = hook_b[3 * layer_idx].input[0].clone().detach() - state[key]["inpactgraddata"]
                        state[key]["wtgradperturb"] = flat_layers[3 * layer_idx + 1].weight.grad.clone().detach() - state[key]["wtgraddata"]
                        state[key]["preactperturb"] = hook_f[3 * layer_idx + 1].input[0].clone().detach() - state[key]["preactdata"]
                        state[key]["preactgradperturb"] = hook_b[3 * layer_idx + 1].input[0].clone().detach() - state[key]["preactgraddata"]
                        state[key]["actperturb"] = hook_f[3 * layer_idx + 2].output.clone().detach() - state[key]["actdata"]
                        state[key]["actgradperturb"] = hook_b[3 * layer_idx + 2].output[0].clone().detach() - state[key]["actgraddata"]
                except Exception:
                    state[key] = {}
            for hook in hook_f + hook_b:
                hook.remove()
            model.zero_grad()

        collect(process_noise=False, perturb=0.0)
        collect(process_noise=True, perturb=0.0)
        collect(process_noise=False, perturb=0.01)
        return state

    def eznas_darts_layer_score(layer_state):
        if not layer_state or "wtgraddata" not in layer_state:
            return 0.0
        # Keep the official EZNAS expression, but run logdet on CPU to avoid
        a = torch.sign(layer_state["wtgraddata"]).detach().cpu()
        try:
            a = torch.logdet(a)
            a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
            a = torch.sigmoid(a)
            a = torch.norm(a.float(), p="fro")
            return float(torch.sum(a).item() / max(1, a.numel()))
        except Exception:
            return None

    n = len(archs) if limit is None else min(limit, len(archs))
    ypred = []
    used_archs = []
    for i, arch in enumerate(archs[:n]):
        model = make_projected_model(seed, device)
        apply_projected_arch(model, arch)
        wrapped = ProjectedTupleNet(model).to(device).train()
        wrapped.apply(initialize_module)
        flat_layers = create_layer_sequence(wrapped, sample)
        state = run_and_collect(wrapped, sample, flat_layers)
        per_layer = [eznas_darts_layer_score(v) for v in state.values()]
        valid_layers = [v for v in per_layer if v is not None]
        score = float(sum(valid_layers) / len(valid_layers)) if valid_layers else -100.0
        ypred.append(score)
        used_archs.append(str(arch))
        del wrapped
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"zcpt_eznas_darts: scored {i + 1}/{n}", flush=True)
    return used_archs, accs[:n], ypred


def make_zcpt_model_scorer(
    method: str,
    seed: int,
    data_root: Path,
    batch_size: int,
    near_repetitions: int = 1,
    epsinas_weights: tuple[float, float] = (0.01, 0.1),
):
    device = torch.device("cuda")

    if method in {"plain", "meco", "zico", "snip", "fisher", "synflow", "grad_norm", "grasp", "jacob_cov", "l2_norm"}:
        ensure_simplejson_shim()
        sys.path.insert(0, str(REPRO_ROOT / "MeCo"))
        sys.path.insert(0, str(REPRO_ROOT / "MeCo" / "zero-cost-nas"))
        from foresight.pruners import predictive
        from foresight.pruners.measures.l2_norm import get_l2_norm_array

        train_queue = zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed)
        inputs = targets = None
        if method in {"l2_norm", "meco"}:
            inputs, targets = first_batch(train_queue, device)

        def score(model: nn.Module) -> float:
            model.train()
            if method == "l2_norm":
                arrays = get_l2_norm_array(model, inputs, targets, mode="param")
                return float(sum(torch.sum(x) for x in arrays).item())
            if method == "meco":
                return float(compute_projected_meco_score(model, inputs))
            model.requires_feature = False
            measures = predictive.find_measures(
                model,
                train_queue,
                ("random", 1, 10),
                device,
                measure_names=[method],
            )
            return scalar_score(measures[method])

        return score

    if method == "nwot":
        inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)

        def score(model: nn.Module) -> float:
            model.train()
            return compute_nwot_score(model, inputs)

        return score

    if method == "flops":
        def score(model: nn.Module) -> float:
            model.eval()
            return compute_flops(model, device)

        return score

    if method == "jacob":
        from sota.cnn.init_projection import Jocab_Score

        inputs, targets = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)

        def score(model: nn.Module) -> float:
            model.eval()
            value = Jocab_Score(model, "normal", inputs, targets, weights=model.get_projected_weights("normal"))
            value += Jocab_Score(model, "reduce", inputs, targets, weights=model.get_projected_weights("reduce"))
            return float(value)

        return score

    if method == "swap":
        sys.path.insert(0, str(REPRO_ROOT / "SWAP"))
        from src.metrics.swap import SWAP
        patch_swap_forward_cpu_activation_concat(SWAP)

        inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)

        def safe_network_weight_gaussian_init(module: nn.Module):
            with torch.no_grad():
                if isinstance(module, nn.Conv2d):
                    if module.weight is not None:
                        nn.init.normal_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    if getattr(module, "weight", None) is not None:
                        nn.init.ones_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    if module.weight is not None:
                        nn.init.normal_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
            return module

        def score(model: nn.Module) -> float:
            model.train()
            swap = SWAP(model=model, inputs=inputs, device=device, seed=seed)
            values = []
            try:
                for _ in range(32):
                    model = model.apply(safe_network_weight_gaussian_init)
                    swap.reinit()
                    values.append(float(swap.forward()))
                    swap.clear()
                result = float(np.mean(values))
            finally:
                if hasattr(swap, "remove_hooks"):
                    swap.remove_hooks()
            del swap, values
            gc.collect()
            torch.cuda.empty_cache()
            return result

        return score

    if method == "near":
        sys.path.insert(0, str(REPRO_ROOT / "NEAR" / "src"))

        loader = zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed)

        def score(model: nn.Module) -> float:
            model = move_projected_state(model, torch.device("cpu"))
            model.train()
            model.requires_grad_(False)
            with torch.no_grad():
                result = float(get_near_score_with_hook_cleanup(model, loader, repetitions=near_repetitions, layer_index=None))
            gc.collect()
            torch.cuda.empty_cache()
            return result

        return score

    if method == "zen":
        ensure_torch_six()
        ensure_torch_compat_apis()
        sys.path.insert(0, str(REPRO_ROOT / "ZenNAS"))
        import ZeroShotProxy.compute_zen_score as zen_score

        def safe_zen_weight_gaussian_init(module: nn.Module):
            with torch.no_grad():
                if isinstance(module, nn.Conv2d):
                    if getattr(module, "weight", None) is not None:
                        nn.init.normal_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    if getattr(module, "weight", None) is not None:
                        nn.init.ones_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    if getattr(module, "weight", None) is not None:
                        nn.init.normal_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
            return module

        zen_score.network_weight_gaussian_init = safe_zen_weight_gaussian_init

        def score(model: nn.Module) -> float:
            adapter = ProjectedDartsFeatureAdapter(model).to(device).train()
            info = zen_score.compute_nas_score(
                gpu=0,
                model=adapter,
                mixup_gamma=1e-2,
                resolution=32,
                batch_size=batch_size,
                repeat=32,
                fp16=False,
            )
            return float(info["avg_nas_score"])

        return score

    if method == "te_nas":
        def score(model: nn.Module) -> float:
            ntk_values = []
            rn_values = []
            for _ in range(3):
                ntk_model = model
                ntk_model.apply(kaiming_normal_fanout_init)
                ntk_values.append(compute_zcpt_ntk_score(ntk_model, 32, batch_size, device))
                rn_model = model
                rn_model.apply(kaiming_normal_fanin_init)
                rn_values.append(compute_zcpt_num_lr(rn_model, 32, batch_size, device))
            return float(np.mean(ntk_values) + np.mean(rn_values))

        return score

    if method == "num_lr":
        def score(model: nn.Module) -> float:
            model.apply(kaiming_normal_fanin_init)
            model.train()
            return compute_zcpt_num_lr(model, 32, batch_size, device)

        return score

    if method == "epe_nas":
        inputs, targets = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)
        official_funcs = load_official_epe_functions(REPRO_ROOT / "EPE-NAS" / "search.py")
        get_batch_jacobian = official_funcs["get_batch_jacobian"]
        eval_score_perclass = official_funcs["eval_score_perclass"]

        class OfficialEpeOutputAdapter(nn.Module):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor):
                return None, self.model(x)

        def score(projected_model: nn.Module) -> float:
            model = OfficialEpeOutputAdapter(projected_model)
            model.train()
            _, logits = model(inputs)
            n_classes = logits.shape[-1]
            jacobs, labels = get_batch_jacobian(model, inputs.detach().clone(), targets, None, None)
            jacobs_np = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
            if len(labels.shape) == 2:
                labels = torch.argmax(labels, dim=1)
            labels_np = labels.cpu().numpy()
            return float(eval_score_perclass(jacobs_np, [labels_np], n_classes))

        return score

    if method == "epsinas":
        inputs, _ = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)

        def prepare_seed():
            random.seed(21)
            np.random.seed(21)
            torch.manual_seed(21)
            torch.cuda.manual_seed(21)

        def initialize_weights(model: nn.Module, constant: float):
            for param in model.parameters():
                nn.init.constant_(param.data, constant)

        def model_forward(model: nn.Module, x: torch.Tensor) -> np.ndarray:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            return out.detach().cpu().numpy()

        def normalize_preds(preds: np.ndarray) -> np.ndarray:
            preds = preds - np.nanmin(preds)
            denom = np.nanmax(preds)
            if np.isfinite(denom) and denom != 0:
                preds = preds / denom
            preds[preds == 0] = np.nan
            return preds

        def score(model: nn.Module) -> float:
            model.train()
            values = []
            for constant in epsinas_weights:
                prepare_seed()
                initialize_weights(model, constant)
                values.append(normalize_preds(model_forward(model, inputs)))
            num = np.nanmean(np.abs(values[0] - values[1]))
            den = np.nanmean(values)
            return float(num / den)

        return score

    if method == "eznas_darts":
        sample = first_batch(zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed), device)[0]

        def initialize_module(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def score(model: nn.Module) -> float:
            wrapped = ProjectedTupleNet(model).to(device).train()
            wrapped.apply(initialize_module)
            flat_layers = create_eznas_layer_sequence(wrapped, sample)
            state = run_eznas_collect(wrapped, sample, flat_layers)
            per_layer = [eznas_darts_layer_score(v) for v in state.values()]
            valid_layers = [v for v in per_layer if v is not None]
            return float(sum(valid_layers) / len(valid_layers)) if valid_layers else -100.0

        return score

    raise ValueError(f"Unsupported op-ablation ZCPT method: {method}")


def score_zcpt_operation_ablation(
    method: str,
    archs,
    accs,
    seed: int,
    data_root: Path,
    limit: int | None,
    batch_size: int,
    near_repetitions: int = 1,
    epsinas_weights: tuple[float, float] = (0.01, 0.1),
):
    device = torch.device("cuda")
    if method == "jacob":
        score_model = make_zcpt_model_scorer(
            method,
            seed,
            data_root,
            batch_size,
            near_repetitions=near_repetitions,
            epsinas_weights=epsinas_weights,
        )
        baseline_model = make_projected_model(seed, device)
        baseline_score = score_model(baseline_model)
        del baseline_model
        torch.cuda.empty_cache()

        records = []
        op_score_map = {}
        for cell_type in ("normal", "reduce"):
            for eid in range(14):
                for opid, op_name in enumerate(SWAP_PRIMITIVES):
                    model = make_projected_model(seed, device)
                    ablate_single_projected_op(model, cell_type, eid, opid)
                    masked_jacob = score_model(model)
                    operation_score = (
                        float(baseline_score - masked_jacob)
                        if np.isfinite(baseline_score) and np.isfinite(masked_jacob)
                        else float("nan")
                    )
                    key = (cell_type, int(eid), int(opid))
                    op_score_map[key] = operation_score
                    records.append(
                        {
                            "cell_type": cell_type,
                            "edge_id": int(eid),
                            "op_id": int(opid),
                            "op_name": op_name,
                            "masked_jacob_score": scalar_score(masked_jacob),
                            "operation_score": scalar_score(operation_score),
                        }
                    )
                    del model
                    torch.cuda.empty_cache()
            print(f"zcpt_{method}: operation ablation finished {cell_type}", flush=True)

        n = len(archs) if limit is None else min(limit, len(archs))
        used_archs = [str(arch) for arch in archs[:n]]
        ypred = aggregate_arch_scores_from_ops(archs[:n], op_score_map)
        return used_archs, accs[:n], ypred, scalar_score(baseline_score), records

    if method == "te_nas":
        te_batches = first_n_batches(
            zcpt_cifar_loader(data_root, batch_size=batch_size, seed=seed, validate_rounds=3),
            device,
            3,
        )

        records = []
        op_score_map = {}
        for cell_type in ("normal", "reduce"):
            origin_model = make_projected_model(seed, device)
            origin_ntk = compute_te_nas_ntk_condition(origin_model, te_batches)
            reset_tenas_rn_seed(seed + 1701)
            origin_rn = compute_zcpt_num_lr(origin_model, 32, batch_size, device)
            del origin_model
            torch.cuda.empty_cache()
            raw_items = []
            for eid in range(14):
                for opid, op_name in enumerate(SWAP_PRIMITIVES):
                    model = make_projected_model(seed, device)
                    ablate_single_projected_op(model, cell_type, eid, opid)
                    ablated_ntk = compute_te_nas_ntk_condition(model, te_batches)
                    reset_tenas_rn_seed(seed + 1701)
                    ablated_rn = compute_zcpt_num_lr(model, 32, batch_size, device)
                    ntk_delta = safe_relative_delta(origin_ntk, ablated_ntk)
                    rn_delta = safe_relative_delta(origin_rn, ablated_rn)
                    raw_items.append(
                        {
                            "cell_type": cell_type,
                            "edge_id": int(eid),
                            "op_id": int(opid),
                            "op_name": op_name,
                            "origin_ntk_condition": scalar_score(origin_ntk),
                            "ablated_ntk_condition": scalar_score(ablated_ntk),
                            "ntk_delta": scalar_score(ntk_delta),
                            "origin_rn_score": scalar_score(origin_rn),
                            "ablated_rn_score": scalar_score(ablated_rn),
                            "rn_delta": scalar_score(rn_delta),
                        }
                    )
                    del model
                    torch.cuda.empty_cache()

            ntk_prune_ranks, rn_prune_ranks = tenas_delta_rank_items(raw_items)

            for idx, item in enumerate(raw_items):
                prune_rank_mean = float((ntk_prune_ranks[idx] + rn_prune_ranks[idx]) / 2.0)
                # TE-NAS projection keeps the masked candidate with the lower
                # NTK/RN rank sum, so the op-ablation aggregate uses the
                # negative rank sum as a max-oriented operation score.
                operation_score = -1.0 * prune_rank_mean
                key = (item["cell_type"], int(item["edge_id"]), int(item["op_id"]))
                op_score_map[key] = operation_score
                item["ntk_prune_rank"] = scalar_score(float(ntk_prune_ranks[idx]))
                item["rn_prune_rank"] = scalar_score(float(rn_prune_ranks[idx]))
                item["te_nas_rank_sum"] = scalar_score(prune_rank_mean)
                item["operation_score"] = scalar_score(operation_score)
                records.append(item)
            print(f"zcpt_{method}: operation ablation finished {cell_type}", flush=True)

        n = len(archs) if limit is None else min(limit, len(archs))
        used_archs = [str(arch) for arch in archs[:n]]
        ypred = aggregate_arch_scores_from_ops(archs[:n], op_score_map)
        baseline_score = None
        return used_archs, accs[:n], ypred, baseline_score, records

    if method in {"l2_norm", "params"}:
        model = make_projected_model(seed, device)
        records = []
        op_score_map = {}
        for cell_type in ("normal", "reduce"):
            for eid in range(14):
                for opid, op_name in enumerate(SWAP_PRIMITIVES):
                    if method == "l2_norm":
                        operation_score = projected_candidate_op_l2_score(model, cell_type, eid, opid)
                    else:
                        operation_score = projected_candidate_op_param_score(model, cell_type, eid, opid)
                    key = (cell_type, int(eid), int(opid))
                    op_score_map[key] = operation_score
                    records.append(
                        {
                            "cell_type": cell_type,
                            "edge_id": int(eid),
                            "op_id": int(opid),
                            "op_name": op_name,
                            "operation_score": scalar_score(operation_score),
                        }
                    )
            print(f"zcpt_{method}: operation ablation finished {cell_type}", flush=True)
        del model
        torch.cuda.empty_cache()

        n = len(archs) if limit is None else min(limit, len(archs))
        used_archs = [str(arch) for arch in archs[:n]]
        ypred = aggregate_arch_scores_from_ops(archs[:n], op_score_map)
        baseline_score = None
        return used_archs, accs[:n], ypred, baseline_score, records

    score_model = make_zcpt_model_scorer(
        method,
        seed,
        data_root,
        batch_size,
        near_repetitions=near_repetitions,
        epsinas_weights=epsinas_weights,
    )
    score_rng_state = capture_rng_state() if method == "swap" else None
    if method == "swap":
        restore_rng_state(score_rng_state)
    baseline_model = make_projected_model(seed, device)
    baseline_score = score_model(baseline_model)
    del baseline_model
    torch.cuda.empty_cache()

    records = []
    op_score_map = {}
    for cell_type in ("normal", "reduce"):
        for eid in range(14):
            for opid, op_name in enumerate(SWAP_PRIMITIVES):
                if method == "swap":
                    restore_rng_state(score_rng_state)
                model = make_projected_model(seed, device)
                ablate_single_projected_op(model, cell_type, eid, opid)
                ablated_score = score_model(model)
                importance = float(baseline_score - ablated_score) if np.isfinite(baseline_score) and np.isfinite(ablated_score) else float("nan")
                key = (cell_type, int(eid), int(opid))
                op_score_map[key] = importance
                records.append(
                    {
                        "cell_type": cell_type,
                        "edge_id": int(eid),
                        "op_id": int(opid),
                        "op_name": op_name,
                        "ablated_score": scalar_score(ablated_score),
                        "operation_score": scalar_score(importance),
                    }
                )
                del model
                torch.cuda.empty_cache()
        print(f"zcpt_{method}: operation ablation finished {cell_type}", flush=True)

    n = len(archs) if limit is None else min(limit, len(archs))
    used_archs = [str(arch) for arch in archs[:n]]
    ypred = aggregate_arch_scores_from_ops(archs[:n], op_score_map)
    return used_archs, accs[:n], ypred, scalar_score(baseline_score), records


def main():
    global MODEL_INIT_CHANNELS, MODEL_LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        choices=[
            "plain",
            "meco",
            "zico",
            "snip",
            "fisher",
            "synflow",
            "grad_norm",
            "grasp",
            "jacob_cov",
            "nwot",
            "jacob",
            "l2_norm",
            "swap",
            "near",
            "zen",
            "te_nas",
            "epe_nas",
            "epsinas",
            "flops",
            "params",
            "num_lr",
            "eznas_darts",
        ],
        required=True,
    )
    ap.add_argument("--arch-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--data-root", default=str(REPRO_ROOT / "data"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epsinas-weights", default="0.01,0.1")
    ap.add_argument("--near-repetitions", type=int, default=1)
    ap.add_argument("--score-mode", choices=["op_ablation", "arch_onehot"], default="op_ablation")
    ap.add_argument("--init-channels", type=int, default=36)
    ap.add_argument("--layers", type=int, default=20)
    args = ap.parse_args()

    MODEL_INIT_CHANNELS = int(args.init_channels)
    MODEL_LAYERS = int(args.layers)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    configure_reproducible_backend(args.method)

    archs, accs = load_pool(args.arch_file)
    data_root = Path(args.data_root)
    invalid_count = None
    epsinas_weights = None
    baseline_score = None
    operation_records = None
    if args.score_mode == "op_ablation":
        if args.method == "epsinas":
            epsinas_weights = tuple(float(x) for x in args.epsinas_weights.split(","))
            if len(epsinas_weights) != 2:
                raise ValueError("--epsinas-weights must contain exactly two comma-separated floats")
        used_archs, ytest, ypred, baseline_score, operation_records = score_zcpt_operation_ablation(
            args.method,
            archs,
            accs,
            args.seed,
            data_root,
            args.limit,
            args.batch_size,
            near_repetitions=args.near_repetitions,
            epsinas_weights=epsinas_weights or (0.01, 0.1),
        )
    else:
        if args.method in {"plain", "meco", "zico", "snip", "fisher", "synflow", "grad_norm", "grasp", "jacob_cov", "l2_norm"}:
            used_archs, ytest, ypred = score_projected_measure_family(args.method, archs, accs, args.seed, data_root, args.limit, args.batch_size)
        elif args.method == "nwot":
            used_archs, ytest, ypred = score_zcpt_nwot(archs, accs, args.seed, data_root, args.limit, args.batch_size)
        elif args.method == "jacob":
            used_archs, ytest, ypred = score_zcpt_jacob(archs, accs, args.seed, data_root, args.limit, args.batch_size)
        elif args.method == "swap":
            used_archs, ytest, ypred = score_zcpt_swap(archs, accs, args.seed, data_root, args.limit, args.batch_size)
        elif args.method == "near":
            used_archs, ytest, ypred = score_zcpt_near(archs, accs, args.seed, data_root, args.limit, args.batch_size, args.near_repetitions)
        elif args.method == "zen":
            used_archs, ytest, ypred = score_zcpt_zen(archs, accs, args.seed, args.limit, args.batch_size)
        elif args.method == "te_nas":
            used_archs, ytest, ypred = score_zcpt_te_nas(archs, accs, args.seed, data_root, args.limit, args.batch_size)
        elif args.method == "num_lr":
            used_archs, ytest, ypred = score_zcpt_num_lr(archs, accs, args.seed, args.limit, args.batch_size)
        elif args.method in {"flops", "params"}:
            used_archs, ytest, ypred = score_zcpt_structural(args.method, archs, accs, args.seed, args.limit)
        elif args.method == "epe_nas":
            used_archs, ytest, ypred, invalid_count = score_zcpt_epe_nas(
                archs, accs, args.seed, data_root, args.limit, args.batch_size
            )
        elif args.method == "epsinas":
            epsinas_weights = tuple(float(x) for x in args.epsinas_weights.split(","))
            if len(epsinas_weights) != 2:
                raise ValueError("--epsinas-weights must contain exactly two comma-separated floats")
            used_archs, ytest, ypred, invalid_count = score_zcpt_epsinas(
                archs, accs, args.seed, data_root, args.limit, args.batch_size, epsinas_weights
            )
        else:
            used_archs, ytest, ypred = score_zcpt_eznas(archs, accs, args.seed, data_root, args.limit, args.batch_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if operation_records is not None:
        write_operation_scores(out_dir, args.method, args.seed, args.batch_size, baseline_score, operation_records)
    payload = [
        {
            "search_space": "nasbench301",
            "dataset": "cifar10",
            "seed": args.seed,
            "predictor": f"official_zcpt_{args.method}",
            "fixed_arch_file": args.arch_file,
            "archs": used_archs,
            "batch_size": args.batch_size,
            "near_repetitions": args.near_repetitions if args.method == "near" else None,
            "epe_nas_source": "VascoLopes/EPE-NAS search.py official get_batch_jacobian/eval_score_perclass runtime adapter" if args.method == "epe_nas" else None,
            "epsinas_weights": list(epsinas_weights) if args.method == "epsinas" else None,
            "invalid_count": invalid_count if args.method == "epe_nas" else None,
            "epsinas_invalid_count": invalid_count if args.method == "epsinas" else None,
            "score_mode": args.score_mode,
            "operation_score_file": "operation_scores.json" if operation_records is not None else None,
            "init_channels": MODEL_INIT_CHANNELS,
            "layers": MODEL_LAYERS,
            "protocol": protocol_label(args.score_mode),
            "protocol_revision": (
                "cell_local_tenas_delta_v1_20260612"
                if args.score_mode == "op_ablation" and args.method == "te_nas"
                else None
            ),
            "network_scale": network_scale_label(),
            "score_stats": score_stats(ypred),
        },
        {
            "full_ytest": ytest,
            "full_testpred": ypred,
            "spearman": corr(ytest, ypred, "spearman"),
            "kendalltau": corr(ytest, ypred, "kendall"),
        },
    ]
    (out_dir / "scores.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out_dir / 'scores.json'}", flush=True)


if __name__ == "__main__":
    main()
