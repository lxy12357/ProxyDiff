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
import torch.nn.functional as F

from . import measure
import sys
from ZeroCostNAS.predictors.pruners.p_utils import get_layer_metric_array


@measure("plain", bn=True, mode="param")
def compute_plain_per_weight(net, inputs, targets, mode, loss_fn, split_data=1):

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    # single batch
    net.zero_grad()
    N = inputs.shape[0]
    grads_abs = 0
    # for sp in range(split_data):
    #     st = sp * N // split_data
    #     en = (sp + 1) * N // split_data

        # outputs = net.forward(inputs[st:en])
        # loss = loss_fn(outputs, targets[st:en])
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = grads_abs+sum_arr(get_layer_metric_array(net, plain, mode))

    # multiple batch
    # N = len(inputs)
    # grads_abs = 0
    # for step in range(N):
    #     net.zero_grad()
    #     input, target = inputs[step].cuda(), targets[step].cuda()
    #     outputs = net.forward(input)
    #     loss = loss_fn(outputs, target)
    #     loss.backward()
    #
    #     # select the gradients that we want to use for search/prune
    #     def plain(layer):
    #         if layer.weight.grad is not None:
    #             return layer.weight.grad * layer.weight
    #         else:
    #             return torch.zeros_like(layer.weight)
    #
    #     grads_abs = grads_abs+sum_arr(get_layer_metric_array(net, plain, mode))
    return grads_abs
