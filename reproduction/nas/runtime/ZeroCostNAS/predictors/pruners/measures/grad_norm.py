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
import logging

import torch
import torch.nn.functional as F

import copy

from . import measure
import sys
from ZeroCostNAS.predictors.pruners.p_utils import get_layer_metric_array
from ZeroCostNAS.utils import utils, setup_logger
import torch.nn as nn

# config = utils.get_config_from_args()
# utils.set_seed(config.seed)
# logger = setup_logger(config.save + "/log.log")
# logger.setLevel(logging.INFO)
# utils.log_args(config)

@measure("grad_norm", bn=True)
def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         logger.info(layer.weight.grad)

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    # single batch
    N = inputs.shape[0]
    grad_norm_arr = 0
    for epoch in range(1):
        # for sp in range(split_data):
        #     st = sp * N // split_data
        #     en = (sp + 1) * N // split_data
        #
        #     outputs = net.forward(inputs[st:en])
        #     loss = loss_fn(outputs, targets[st:en])
        outputs = net.forward(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        grad_norm_arr = grad_norm_arr+sum_arr(get_layer_metric_array(
            net,
            lambda l: l.weight.grad.norm()
            if l.weight.grad is not None
            else torch.zeros_like(l.weight),
            mode="param",
        ))

    # multiple batch
    # N = len(inputs)
    # grad_norm_arr = 0
    # for step in range(N):
    #     input, target = inputs[step].cuda(), targets[step].cuda()
    #     for layer in net.modules():
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             logger.info(layer.weight.grad)
    #     net.zero_grad()
    #     outputs = net(input)
    #     loss = loss_fn(outputs, target)
    #     loss.backward()
    #
    #     grad_norm_arr = grad_norm_arr + sum_arr(get_layer_metric_array(
    #         net,
    #         lambda l: l.weight.grad.norm()
    #         if l.weight.grad is not None
    #         else torch.zeros_like(l.weight),
    #         mode="param",
    #     ))

    return grad_norm_arr
