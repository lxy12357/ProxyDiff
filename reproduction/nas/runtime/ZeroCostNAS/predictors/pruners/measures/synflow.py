# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch

from . import measure
import sys
from ZeroCostNAS.predictors.pruners.p_utils import get_layer_metric_array


@measure("synflow", bn=False, mode="param")
@measure("synflow_bn", bn=True, mode="param")
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()
    device = inputs.device
    grads_abs = 0

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net.double())

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    # print("input_dim: "+str(input_dim))
    # inputs = torch.ones([1] + input_dim).to(device)

    inputs = torch.ones([1] + input_dim).double().to(device)
    # print("inputs: " + str(inputs))

    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = grads_abs+sum_arr(get_layer_metric_array(net, synflow, mode))

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs
