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
import torch.nn as nn
import torch.nn.functional as F
import types
import copy

from .p_utils import *
from . import measures
from .measures.model_stats import get_model_stats
from ZeroCostNAS.utils import utils, setup_logger

# config = utils.get_config_from_args()
# utils.set_seed(config.seed)
# logger = setup_logger(config.save + "/log.log")
# logger.setLevel(logging.INFO)
# utils.log_args(config)


def no_op(self, x):
    return x


def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn == False:
        for l in net.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    return net


def find_measures_arrays(
    net_orig,
    trainloader,
    data,
    dataload_info,
    device,
    measure_names=None,
    loss_fn=F.cross_entropy,
    epoch_num = 1,
    epoch_by_epoch = False,
):
    if measure_names is None:
        measure_names = measures.available_measures

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig, "get_prunable_copy"):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    # move to cpu to free up mem
    torch.cuda.empty_cache()
    # net_orig = net_orig.cpu()
    # torch.cuda.empty_cache()

    # given 1 minibatch of data

    measure_values = {}
    for measure_name in measure_names:
        if epoch_by_epoch:
            measure_values[measure_name] = []
            val_tmp = 0
        else:
            measure_values[measure_name] = 0
    # multiple epoch
    # logger.info("epoch_num: "+str(epoch_num))
    for epoch in range(epoch_num):
        if dataload == "random":
            if data:
                inputs, targets = data[0],data[1]
            else:
                inputs, targets = get_some_data(
                    trainloader, num_batches=num_imgs_or_batches, device=device
                )
        elif dataload == "grasp":
            inputs, targets = get_some_data_grasp(
                trainloader,
                num_classes,
                samples_per_class=num_imgs_or_batches,
                device=device,
            )
        else:
            raise NotImplementedError(f"dataload {dataload} is not supported")

        done, ds = False, 1

        while not done:
            try:
                for measure_name in measure_names:
                    # if measure_name not in measure_values:
                    net_orig = net_orig.clone()
                    val = measures.calc_measure(
                        measure_name,
                        net_orig,
                        device,
                        inputs,
                        targets,
                        loss_fn=loss_fn,
                        split_data=ds,
                    )
                    if epoch_by_epoch:
                        val_tmp = val_tmp + val
                        measure_values[measure_name].append(val_tmp)
                    else:
                        measure_values[measure_name] = measure_values[measure_name]+val
                    # logger.info("val:" + str(val))
                    # logger.info("sum:"+str(measure_values[measure_name]))

                done = True
            except RuntimeError as e:
                if "out of memory" in str(e):
                    done = False
                    if ds == inputs.shape[0] // 2:
                        raise ValueError(
                            f"Can't split data anymore, but still unable to run. Something is wrong"
                        )
                    ds += 1
                    while inputs.shape[0] % ds != 0:
                        ds += 1
                    torch.cuda.empty_cache()
                    print(f"Caught CUDA OOM, retrying with data split into {ds} parts")
                else:
                    raise e

    net_orig = net_orig.to(device).train()
    return measure_values


def find_measures(
    net_orig,  # neural network
    dataloader,  # a data loader (typically for training data)
    data,
    dataload_info,  # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
    device,  # GPU/CPU device used
    loss_fn,  # loss function to use within the zero-cost metrics
    measure_names=None,  # an array of measure names to compute, if left blank, all measures are computed by default
    measures_arr=None,
    epoch_num = 1,
    epoch_by_epoch = False,
):

    # Given a neural net
    # and some information about the input data (dataloader)
    # and loss function (loss_fn)
    # this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    measure_score = []
    if measure_names[0] in ['flops', 'params']:
        if data:
            x, target = data[0], data[1]
        else:
            data_iterator = iter(dataloader)
            x, target = next(data_iterator)
        x_shape = list(x.shape)
        x_shape[0] = 1 # to prevent overflow

        model_stats = get_model_stats(
            net_orig,
            input_tensor_shape=x_shape,
            clone_model=True
        )

        if measure_names[0] == 'flops':
            measure_score_tmp = float(model_stats.Flops)/1e6 # megaflops
        else:
            measure_score_tmp = float(model_stats.parameters)/1e6 # megaparams
        # return measure_score
        measure_score.append(measure_score_tmp)
        if len(measure_names)>1:
            measure_names = measure_names[1:]
        else:
            measure_names = []
        print(measure_names)

    if measures_arr is None:
        measures_arr = find_measures_arrays(
            net_orig,
            dataloader,
            data,
            dataload_info,
            device,
            loss_fn=loss_fn,
            measure_names=measure_names,
            epoch_num = epoch_num,
            epoch_by_epoch = epoch_by_epoch
        )

    for k, v in measures_arr.items():
        # single batch
        # if k == "jacov" or k == 'epe_nas' or k=='nwot' or k=='zen':
        #     measure_score = v
        # else:
        #     measure_score = sum_arr(v)
        # multiple batch
        measure_score.append(v)
    return measure_score
