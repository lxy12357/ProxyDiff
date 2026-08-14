#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "src"))
sys.argv = [
    sys.argv[0],
    "--config-file",
    str(ROOT / "configs" / "nb301_proxy_refinement.yaml"),
]

import _proxydiff_nb301_refinement_impl as refinement


class FakeCell:
    def __init__(self, name):
        self.name = name
        self.received = None

    def forward(self, x, metric_para=None, **kwargs):
        self.received = metric_para
        return x


class FakeSearchSpace:
    def __init__(self):
        self.num_edges = 2
        self.num_ops = 4
        self.metric_num = 3
        self.metric_epoch_accumulate = [False] * self.metric_num
        self.has_axis_calib = True
        self.has_component_corr = True
        self.axis_calibration_weights = [torch.zeros(self.metric_num, requires_grad=True)]
        self.component_corr_weights = [
            torch.zeros(self.num_edges, self.num_ops - 1, requires_grad=True)
            for _ in range(2)
        ]
        generator = torch.Generator().manual_seed(9000)
        self._score = [
            [[torch.randn(self.num_edges, self.num_ops - 1, generator=generator) for _ in range(2)]]
            for _ in range(self.metric_num)
        ]
        self.cells = [FakeCell("normal_cell"), FakeCell("reduction_cell")]
        self.nodes = {
            0: {"subgraph": self.cells[0]},
            1: {"subgraph": self.cells[1]},
        }

    def forward(self, x, **kwargs):
        legacy = torch.zeros(self.num_edges, self.num_ops - 1)
        for cell in self.cells:
            cell.forward(x, metric_para=legacy)
        return x


def assert_close(left, right, tolerance=1e-6):
    if not torch.allclose(left, right, atol=tolerance, rtol=0.0):
        raise AssertionError(float((left - right).abs().max()))


def main():
    tie_space = SimpleNamespace(
        num_ops=4,
        _masks=[torch.zeros(2, 3), torch.zeros(2, 3)],
    )
    tie_score = torch.zeros(2, 2, 3)
    tie_masks = refinement._apply_nb301_topk_mask(tie_space, tie_score, topk=2)
    expected_tie_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    for mask in tie_masks:
        assert_close(mask, expected_tie_mask)

    os.environ["PROXYDIFF_METRIC_WEIGHT_STYLE"] = "latent_fixed_signed_residual"
    search_space = FakeSearchSpace()
    raw = refinement._build_nb301_proxydiff_cell_scores(search_space)
    transported = refinement._rank_equivalent_forward_scores(raw)

    # Every legal NB301 architecture contributes eight operations per cell, so
    # the shared affine transport must preserve all architecture-score orders.
    generator = torch.Generator().manual_seed(9000)
    raw_totals = []
    transported_totals = []
    for _ in range(128):
        raw_total = torch.zeros(())
        transported_total = torch.zeros(())
        for cell in range(2):
            indices = torch.randint(0, raw[cell].numel(), (8,), generator=generator)
            raw_total += raw[cell].reshape(-1)[indices].sum()
            transported_total += transported[cell].reshape(-1)[indices].sum()
        raw_totals.append(raw_total.detach())
        transported_totals.append(transported_total.detach())
    raw_order = torch.argsort(torch.stack(raw_totals))
    transported_order = torch.argsort(torch.stack(transported_totals))
    if not torch.equal(raw_order, transported_order):
        raise AssertionError("global affine transport changed architecture ordering")

    probe = [torch.randn_like(cell) for cell in raw]
    raw_loss = sum((cell * weight).sum() for cell, weight in zip(raw, probe))
    raw_grad = torch.autograd.grad(
        raw_loss, search_space.axis_calibration_weights, retain_graph=True
    )[0]
    transported_loss = sum(
        (cell * weight).sum() for cell, weight in zip(transported, probe)
    )
    transported_grad = torch.autograd.grad(
        transported_loss, search_space.axis_calibration_weights
    )[0]
    flat = torch.cat([cell.reshape(-1) for cell in raw])
    span = flat.detach().max() - flat.detach().min()
    mapped_flat = torch.cat([cell.reshape(-1) for cell in transported])
    operation_count = int(raw[0].shape[-1])
    assert_close(mapped_flat.mean() * operation_count, torch.ones(()))
    unscaled = [(cell - flat.detach().min()) / span + torch.finfo(flat.dtype).eps for cell in raw]
    mass = (
        torch.cat([cell.reshape(-1) for cell in unscaled]).detach().mean()
        * operation_count
    )
    scale = 1.0 / (span * mass)
    assert_close(transported_grad, raw_grad * scale)

    os.environ["PROXYDIFF_COMPONENT_CORR_LINK"] = "unit_slope_bounded"
    component_probe = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    correction = refinement._component_correction(component_probe)
    assert_close(correction, torch.tanh(component_probe))
    correction.sum().backward()
    assert_close(component_probe.grad[1], torch.ones(()))

    os.environ["PROXYDIFF_COMPONENT_CORR_CENTERING"] = "within_context_zero_mean"
    centered_probe = torch.tensor(
        [[-2.0, 0.0, 2.0], [1.0, 2.0, 4.0]], requires_grad=True
    )
    centered = refinement._component_correction(centered_probe)
    assert_close(centered.mean(dim=-1), torch.zeros(2))
    del os.environ["PROXYDIFF_COMPONENT_CORR_CENTERING"]

    aligned = refinement._mean_unit_gradient(
        [torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0])]
    )
    assert_close(aligned, torch.tensor([1.0, 0.0]))
    opposed = refinement._mean_unit_gradient(
        [torch.tensor([2.0, 0.0]), torch.tensor([-1.0, 0.0])]
    )
    assert_close(opposed, torch.zeros(2))
    partial = refinement._mean_unit_gradient(
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    )
    assert_close(partial, torch.tensor([0.5, 0.5]))

    refinement.install_nb301_canonical_score_forward(search_space)
    search_space.forward(torch.ones(1))
    for cell_index, cell in enumerate(search_space.cells):
        assert_close(cell.received, transported[cell_index])

    print("NB301 canonical forward parity: PASS")


if __name__ == "__main__":
    main()
