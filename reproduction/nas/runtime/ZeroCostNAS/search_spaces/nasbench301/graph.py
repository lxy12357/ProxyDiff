import os
import random
import torch
import logging
import numpy as np
import networkx as nx
import pickle

from collections import namedtuple

from networkx import lexicographical_topological_sort
from torch import nn
from copy import deepcopy
from ConfigSpace.read_and_write import json as config_space_json_r_w
from torch.autograd import Variable

from .primitives import FactorizedReduce
import sys


from ZeroCostNAS.search_spaces.core import primitives as ops
from ZeroCostNAS.search_spaces.core.graph import Graph, EdgeData
from ZeroCostNAS.search_spaces.nasbench301.conversions import (
    convert_compact_to_naslib,
    convert_naslib_to_compact,
    convert_naslib_to_genotype,
    make_compact_mutable,
    make_compact_immutable,
)
from ZeroCostNAS.search_spaces.core.query_metrics import Metric
from ZeroCostNAS.search_spaces.core.primitives import AbstractCombOp, AbstractPrimitive, ReLUConvBN
from ZeroCostNAS.utils.utils import get_project_root, AttrDict, drop_path
from ZeroCostNAS.utils.logging import log_formats, setup_logger

import torch.nn.functional as F

logger = logging.getLogger(__name__)

# NUM_VERTICES = 4
# NUM_OPS = 7


class NasBench301SearchSpace(Graph):
    """
    The search space for CIFAR-10 as defined in

        Liu et al. 2019: DARTS: Differentiable Architecture Search

    It consists of a makrograph which is predefined and not optimized
    and two kinds of learnable cells: normal and reduction cells. At
    each edge are 8 primitive operations.
    """

    """
    Scope is used to target different instances of the same cell.
    Here we divide the cells in normal/reduction cell and stage.
    This is necessary to set the correct channels at each stage.
    The architecture optimizer should consider all of them equally.
    """
    OPTIMIZER_SCOPE = [
        "n_stage_1",
        "n_stage_2",
        "n_stage_3",
        "r_stage_1",
        "r_stage_2",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, in_channels=3, auxiliary=False,metric_names=[], metric_epoch_accumulate=[]):
        """
        Initialize a new instance of the DARTS search space.
        Note:
            __init__ cannot take any parameters due to the way networkx is implemented.
            If we want to change the number of classes set a static attribute `NUM_CLASSES`
            before initializing the class. Default is 10 as for cifar-10.
        """
        super().__init__()

        self.channels = [32, 64, 128]
        # self.channels = [16, 32, 64]
        self.compact = None
        self.load_labeled = None
        self.num_classes = n_classes
        self.max_epoch = 100
        self.in_channels = in_channels
        self.space_name = "nasbench301"
        self.auxiliary_output = auxiliary
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        self.num_vertices = 4
        self.num_edges = 14
        self.num_ops = 8
        self.op_names = ["MaxPool","AvgPool",("Identity", "FactorizedReduce"),"SepConv3x3","SepConv5x5","DilConv3x3","DilConv5x5","Zero"]

        self.metric_names = metric_names
        self.metric_num = len(metric_names)
        self.has_metric_para = True
        if metric_epoch_accumulate != []:
            self.metric_epoch_accumulate = metric_epoch_accumulate
        else:
            self.metric_epoch_accumulate = [False for m in range(self.metric_num)]
        self.has_axis_calib = True
        self.has_component_corr = True
        # logger.info("metric_epoch_accumulate: " + str(self.metric_epoch_accumulate))
        self._score = [[[torch.ones(self.num_edges, self.num_ops-1).cuda() for c in range(2)] for e in range(11)] for m in range(self.metric_num)]
        self.axis_calibration_weights = [Variable(torch.zeros(self.metric_num).cuda(), requires_grad=True)]
        self.edge_epoch_weights = [Variable(torch.zeros(11).cuda(), requires_grad=True) for m in range(self.metric_num)]
        self.component_corr_weights = [Variable(torch.zeros(self.num_edges, self.num_ops-1).cuda(), requires_grad=True) for c in range(2)]
        self._masks = [torch.ones(self.num_edges, self.num_ops-1).cuda() for c in range(2)]

        self.layers = nn.Sequential(
            nn.Linear(self.metric_num,8,bias=False),
            # nn.ReLU(),
            nn.Linear(8,self.metric_num),
            # nn.Sigmoid()
            nn.Softmax()
        )

        # print(self.layers[0].weight)
        # # print(self.layers[0].bias)
        #
        # def init_weights(m):
        #     if type(m)==nn.Linear:
        #         nn.init.constant(m.weight,1)
        #
        #
        # self.layers.apply(init_weights)

        """
        Build the search space with the parameters specified in __init__.
        """
        #
        # Cell definition
        #

        # Normal cell first
        normal_cell = Graph()
        normal_cell.name = (
            "normal_cell"  # Use the same name for all cells with shared attributes
        )

        # Input nodes
        normal_cell.add_node(1)
        normal_cell.add_node(2)

        # Intermediate nodes
        normal_cell.add_node(3)
        normal_cell.add_node(4)
        normal_cell.add_node(5)
        normal_cell.add_node(6)

        # Output node
        normal_cell.add_node(7)

        # Edges
        normal_cell.add_edges_from([(1, i) for i in range(3, 7)])  # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, 7)])  # input 2
        normal_cell.add_edges_from([(3, 4), (3, 5), (3, 6)])
        normal_cell.add_edges_from([(4, 5), (4, 6)])
        normal_cell.add_edges_from([(5, 6)])

        self.edge_to_inout = {}
        for i in range(2):
            self.edge_to_inout[i] = (i,2)
        for i in range(3):
            self.edge_to_inout[i+2] = (i, 3)
        for i in range(4):
            self.edge_to_inout[i + 5] = (i, 4)
        for i in range(5):
            self.edge_to_inout[i + 9] = (i, 5)

        self.inout_to_edge = {}
        for i in range(2):
            self.inout_to_edge[(i,2)] = i
        for i in range(3):
            self.inout_to_edge[(i, 3)] = i+2
        for i in range(4):
            self.inout_to_edge[(i, 4)] = i+5
        for i in range(5):
            self.inout_to_edge[(i, 5)] = i+9

        # node start -> node end
        # (0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5) ->
        # (0,2),(1,2),(0,3),(1,3),(2,3),(0,4),(1,4),(2,4),(3,4),(0,5),(1,5),(2,5),(3,5),(4,5)
        self.edge_mapping = {0:0,1:2,2:5,3:9,4:1,5:3,6:6,7:10,8:4,9:7,10:11,11:8,12:12,13:13}

        # Edges connecting to the output are always the identity
        normal_cell.add_edges_from(
            [(i, 7, EdgeData().finalize()) for i in range(3, 7)]
        )  # output

        # Reduction cell has the same topology
        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        # set the cell name for all edges. This is necessary to convert a genotype to a naslib object
        for _, _, edge_data in normal_cell.edges.data():
            if not edge_data.is_final():
                edge_data.set("cell_name", "normal_cell")

        for _, _, edge_data in reduction_cell.edges.data():
            if not edge_data.is_final():
                edge_data.set("cell_name", "reduction_cell")

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)  # input node
        self.add_node(2)  # preprocessing
        self.add_node(3)

        # cells
        normal_cell.set_op_names(self.op_names)
        reduction_cell.set_op_names(self.op_names)

        self.add_node(4, subgraph=normal_cell.set_scope("n_stage_1").set_input([2, 3]))
        normal_cell_copy1 = normal_cell.copy()
        normal_cell_copy1.set_op_names(self.op_names)
        self.add_node(
            5, subgraph=normal_cell_copy1.set_scope("n_stage_1").set_input([2, 4])
        )
        self.add_node(
            6, subgraph=reduction_cell.set_scope("r_stage_1").set_input([4, 5])
        )
        normal_cell_copy2 = normal_cell.copy()
        normal_cell_copy2.set_op_names(self.op_names)
        self.add_node(
            7, subgraph=normal_cell_copy2.set_scope("n_stage_2").set_input([5, 6])
        )
        normal_cell_copy3 = normal_cell.copy()
        normal_cell_copy3.set_op_names(self.op_names)
        self.add_node(
            8, subgraph=normal_cell_copy3.set_scope("n_stage_2").set_input([6, 7])
        )
        reduction_cell_copy1 = reduction_cell.copy()
        reduction_cell_copy1.set_op_names(self.op_names)
        self.add_node(
            9, subgraph=reduction_cell_copy1.set_scope("r_stage_2").set_input([7, 8])
        )
        normal_cell_copy4 = normal_cell.copy()
        normal_cell_copy4.set_op_names(self.op_names)
        self.add_node(
            10, subgraph=normal_cell_copy4.set_scope("n_stage_3").set_input([8, 9])
        )
        normal_cell_copy5 = normal_cell.copy()
        normal_cell_copy5.set_op_names(self.op_names)
        self.add_node(
            11, subgraph=normal_cell_copy5.set_scope("n_stage_3").set_input([9, 10])
        )

        # output
        self.add_node(12)

        # chain connections
        self.add_edges_from([(i, i + 1) for i in range(1, 11)])

        # skip connections
        self.add_edges_from([(i, i + 2) for i in range(4, 10)])
        self.add_edge(2, 4)
        self.add_edge(2, 5)

        if self.auxiliary_output:
            # node 12 becomes aux head
            self.add_node(13)

            # auxiliary
            self.add_edge(11, 12)

            # final output
            self.add_edge(11, 13)
        else:
            # final output
            self.add_edge(11, 12)

        #
        # Operations at the makrograph edges
        #
        self.num_in_edges = 4
        reduction_cell_indices = [6, 9]

        channel_map_from, channel_map_to = channel_maps(
            reduction_cell_indices, max_index=12
        )

        self._set_makrograph_ops(
            channel_map_from,
            channel_map_to,
            reduction_cell_indices,
            max_index=12,
            affine=True,
        )

        self._set_cell_ops(reduction_cell_indices)

        self.prepare_evaluation()

    def projection_step(self,m):
        if type(m) == nn.Linear:
            m.weight.data = m.weight.data.clamp(min=0)

    def _set_makrograph_ops(
        self,
        channel_map_from,
        channel_map_to,
        reduction_cell_indices,
        max_index,
        affine=True,
    ):
        # pre-processing
        # In darts there is a hardcoded multiplier of 3 for the output of the stem
        stem_multiplier = 3
        self.edges[1, 2].set("op", ops.Stem(C_in=self.in_channels,
                                            C_out=self.channels[0] * stem_multiplier))

        # edges connecting cells
        for u, v, data in sorted(self.edges(data=True)):
            if u > 1 and v < max_index:
                if u == 3:
                    continue
                C_in = self.channels[channel_map_from[u]]
                C_out = self.channels[channel_map_to[v]]
                if C_in == C_out:
                    C_in = (
                        C_in * stem_multiplier if u == 2 else C_in * self.num_in_edges
                    )  # handle Stem
                    if v in reduction_cell_indices:
                        C_out *= 2
                    data.set(
                        "op", ops.ReLUConvBN(C_in, C_out, kernel_size=1, affine=affine)
                    )
                else:
                    data.set(
                        "op",
                        FactorizedReduce(
                            C_in * self.num_in_edges, C_out, affine=affine
                        ),
                    )

        # post-processing
        _, _, data = sorted(self.edges(data=True))[-1]
        data.set(
            "op",
            ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1] * self.num_in_edges, self.num_classes),
            ),
        )

    def _set_cell_ops(self, reduction_cell_indices):
        # normal cells
        stages = ["n_stage_1", "n_stage_2", "n_stage_3"]

        for scope, c in zip(stages, self.channels):
            self.update_edges(
                update_func=lambda edge: NasBench301SearchSpace._set_ops(edge, c, stride=1),
                scope=scope,
                private_edge_data=True,
            )

        # reduction cells
        # stride=2 is only for some edges, that's why we have to do it this way
        for n, c in zip(reduction_cell_indices, self.channels[1:]):
            reduction_cell = self.nodes[n]["subgraph"]
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                if not data.is_final():
                    edge = AttrDict(data=data)
                    NasBench301SearchSpace._set_ops(edge, c, stride)

        #
        # Combining operations
        #
        for _, cell in sorted(self.nodes("subgraph")):
            if cell:
                cell.nodes[7]["comb_op"] = channel_concat

    @staticmethod
    def _set_ops(edge, C, stride):
        """
        Replace the 'op' at the edges with the ones defined here.
        This function is called by the framework for every edge in
        the defined scope.
        Args:
            current_egde_data (EdgeData): The data that currently sits
                at the edge.
            C (int): convolutional channels
            stride (int): stride for the operation

        Returns:
            EdgeData: the updated EdgeData object.
        """
        edge.data.set(
            "op",
            [
                ops.Identity()
                if stride == 1
                else FactorizedReduce(C, C, stride, affine=False),
                ops.Zero(stride=stride),
                ops.MaxPool(C, 3, stride, use_bn=True),
                ops.AvgPool(C, 3, stride, use_bn=True),
                ops.SepConv(C, C, kernel_size=3, stride=stride, padding=1, affine=False),
                ops.SepConv(C, C, kernel_size=5, stride=stride, padding=2, affine=False),
                ops.DilConv(
                    C, C, kernel_size=3, stride=stride, padding=2, dilation=2, affine=False
                ),
                ops.DilConv(
                    C, C, kernel_size=5, stride=stride, padding=4, dilation=2, affine=False
                ),
            ],
        )

    def forward(self, x, drop_prob=0, epoch=0, to_print=False, fix_logger=None,no_edge_normalize=False, *args):
        """
        Forward some data through the graph. This is done recursively
        in case there are graphs defined on nodes or as 'op' on edges.

        Args:
            x (Tensor or dict): The input. If the graph sits on a node the
                input can be a dict with {source_idx: Tensor} to be routed
                to the defined input nodes. If the graph sits on an edge,
                x is the feature tensor.
            args: This is only required to handle cases where the graph sits
                on an edge and receives an EdgeData object which will be ignored
        """
        logger.debug("Graph {} called. Input {}.".format(self.name, log_formats(x)))

        if to_print:
            new_logger = setup_logger(fix_logger + "/new_log.log")
            new_logger.setLevel(logging.INFO)
            # new_logger.info("score: {}".format(metric_para))
            new_logger.info("input_init: {}".format(x))

        # Assign x to the corresponding input nodes
        self._assign_x_to_nodes(x)

        for node_idx in lexicographical_topological_sort(self):
            # logger.info("node_idx: "+str(node_idx))
            node = self.nodes[node_idx]
            logger.debug(
                "Node {}-{}, current data {}, start processing...".format(
                    self.name, node_idx, log_formats(node)
                )
            )
            if to_print:
                new_logger.info("Node {}-{}, current data {}, start processing...".format(
                    self.name, node_idx, log_formats(node)
                ))
            # node internal: process input if necessary
            # if ("subgraph" in node and "comb_op" not in node) or (
            # "comb_op" in node and "subgraph" not in node
            # ):
            # log_first_n(
            # logging.WARN, "Comb_op is ignored if subgraph is defined!", n=1
            # )
            # TODO: merge 'subgraph' and 'comb_op'. It is basicallly the same thing. Also in parse()
            if "subgraph" in node:
                # logger.info("subgraph")
                if node["subgraph"].name == "normal_cell":
                    cell = 0
                elif node["subgraph"].name == "reduction_cell":
                    cell = 1
                if self.has_metric_para:
                    metric_para = None
                else:
                    # print("here")
                    # if epoch==0:
                    #     if isinstance(edge_data.op,ReLUConvBN):
                    #         print(edge_data.op.op[1].weight)
                    #         sys.exit()
                    metric_para = None
                if to_print:
                    new_logger.info("cell:" + str(cell))
                    new_logger.info("mask:"+str(self._masks[cell]))
                x = node["subgraph"].forward(node["input"],metric_para=metric_para,edge_mapping = self.edge_mapping,mask=self._masks[cell],no_edge_normalize=no_edge_normalize,to_print=to_print, fix_logger=fix_logger)
                if to_print:
                    new_logger.info("after x:"+str(x))
            else:
                if to_print:
                    new_logger.info("b")
                if len(node["input"].values()) == 1:
                    x = list(node["input"].values())[0]
                    # if to_print:
                    #     new_logger.info("input_after: {}".format(x))
                    if to_print:
                        new_logger.info("c")
                else:
                    if to_print:
                        new_logger.info("d")
                    comb_op = node["comb_op"]
                    input_tensors = [node["input"][k] for k in sorted(node["input"].keys())]

                    if isinstance(comb_op, AbstractCombOp):
                        in_edges = [self.get_edge_data(u, v) for u, v in sorted(self.in_edges(node_idx), key=lambda x: x[0])]
                        if no_edge_normalize:
                            x = input_tensors
                        else:
                            x = comb_op(input_tensors, in_edges)
                    else:
                        # logger.info("before outer sum:" + str(input_tensors))
                        if comb_op == sum:
                            if no_edge_normalize:
                                x = comb_op(input_tensors)
                            else:
                                x = comb_op(input_tensors) / len(input_tensors)
                        else:
                            x = comb_op(input_tensors)

            node["input"] = {}  # clear the input as we have processed it

            if (
                    len(list(self.neighbors(node_idx))) == 0
                    and node_idx < list(lexicographical_topological_sort(self))[-1]
            ):
                # We have more than one output node. This is e.g. the case for
                # auxillary losses. Attach them to the graph, handling must done
                # by the user.
                logger.debug(
                    "Graph {} has more then one output node. Storing output of non-maximum index node {} at graph dict".format(
                        self, node_idx
                    )
                )
                self.graph["out_from_{}".format(node_idx)] = x
                if to_print:
                    new_logger.info("g")
            else:
                if to_print:
                    new_logger.info("h")
                # outgoing edges: process all outgoing edges
                for neigbor_idx in self.neighbors(node_idx):
                    edge_data = self.get_edge_data(node_idx, neigbor_idx)
                    # inject edge data only for AbstractPrimitive, not Graphs
                    if isinstance(edge_data.op, Graph):
                        if to_print:
                            new_logger.info("1")
                        # logger.info("graph")
                        # if to_print:
                        #     new_logger.info("input: {}".format(x))
                        edge_output = edge_data.op.forward(x, to_print=to_print, fix_logger=fix_logger)
                        # if to_print:
                        #     new_logger.info("output: {}".format(edge_output))
                    # elif isinstance(edge_data.op, list):
                    #     # logger.info("before inner sum:"+str(x))
                    #     # logger.info("list")
                    #     try:
                    #         edge_output = drop_path(edge_data.op[0].forward(x, edge_data=edge_data), drop_prob)
                    #         for primitive in edge_data.op[1:]:
                    #             # if not isinstance(primitive, Zero()):
                    #             edge_output = edge_output + drop_path(primitive.forward(x, edge_data=edge_data),
                    #                                                   drop_prob)
                    #         # edge_output = edge_output / (len(edge_data.op))
                    #     except Exception as e:
                    #         logger.info("edge_data:" + str(edge_data))
                    #     # logger.info("after inner sum:" + str(edge_output))
                    elif isinstance(edge_data.op, AbstractPrimitive):
                        if to_print:
                            new_logger.info("2")
                        # logger.info("AbstractPrimitive")
                        logger.debug(
                            "Processing op {} at edge {}-{}".format(
                                edge_data.op, node_idx, neigbor_idx
                            )
                        )
                        edge_output = drop_path(edge_data.op.forward(x, edge_data=edge_data), drop_prob)
                        # if to_print:
                        #     new_logger.info("output: "+str(edge_output))
                    else:
                        if to_print:
                            new_logger.info("3")
                        raise ValueError(
                            "Unknown class as op: {}. Expected either Graph or AbstactPrimitive".format(
                                edge_data.op
                            )
                        )
                    self.nodes[neigbor_idx]["input"].update({node_idx: edge_output})

            logger.debug("Node {}-{}, processing done.".format(self.name, node_idx))

        logger.debug("Graph {} exiting. Output {}.".format(self.name, log_formats(x)))
        return x

    def calculate_param(self):

        total_params = 0
        for node_idx in lexicographical_topological_sort(self):
            node = self.nodes[node_idx]
            if "subgraph" in node:
                total_params += node["subgraph"].calculate_param()

            for neigbor_idx in self.neighbors(node_idx):
                edge_data = self.get_edge_data(node_idx, neigbor_idx)
                if isinstance(edge_data.op, Graph):
                    total_params += self.calculate_param(edge_data.op)
                elif isinstance(edge_data.op, list):
                    for op_tmp in edge_data.op:
                        total_params += np.sum(np.prod(v.size()) for name, v in op_tmp.named_parameters())
                else:
                    total_params += np.sum(np.prod(v.size()) for name, v in edge_data.op.named_parameters())
        return total_params/ 1e6


    def prepare_discretization(self):
        """
        In DARTS a node can have a maximum of two incoming edges.
        This is handled here.
        """

        self.update_nodes(
            NasBench301SearchSpace._truncate_input_edges, scope=self.OPTIMIZER_SCOPE,
            single_instances=True
        )

    def prepare_evaluation(self):
        """
        In DARTS the evaluation model has 32 channels after the Stem
        and 3 normal cells at each stage.
        """

        # Taken from DARTS implementation
        # assuming input size 8x8
        if self.auxiliary_output:
            self.edges[11, 12].set(
                "op",
                ops.Sequential(
                    nn.ReLU(inplace=False),
                    nn.AvgPool2d(
                        5, stride=3, padding=0, count_include_pad=False
                    ),  # image size = 2 x 2
                    nn.Conv2d(self.channels[-1] * self.num_in_edges, 128, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(128, 768, 2, bias=False),
                    nn.BatchNorm2d(768),
                    nn.ReLU(inplace=False),
                    nn.Flatten(),
                    nn.Linear(768, self.num_classes),
                ),
            )

    def auxiliary_logits(self):
        return self.graph["out_from_12"]

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nasbench 301
        """
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

        metric_to_nb301 = {
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_ACCURACY: "val_accuracies",
            Metric.TEST_ACCURACY: "val_accuracies",
            Metric.TRAIN_TIME: "runtime",
        }

        assert not epoch or epoch in [-1, 100]
        # assert metric in [Metric.VAL_ACCURACY, Metric.RAW]
        genotype = convert_naslib_to_genotype(self)
        if metric == Metric.VAL_ACCURACY:
            val_acc = dataset_api["nb301_model"][0].predict(
                config=genotype, representation="genotype"
            )
            return val_acc
        elif metric == Metric.TRAIN_TIME:
            runtime = dataset_api["nb301_model"][1].predict(
                config=genotype, representation="genotype"
            )
            return runtime
        else:
            return -1

    @staticmethod
    def _truncate_input_edges(node, in_edges, out_edges):
        """
        Removes input edges if there are more than k.
        """

        def _largest_post_softmax_weight(edge) -> int:
            _, edge_data = edge

            alpha = edge_data.alpha.detach()
            # The zero operation has its value set to -inf to ensure it never gets selected
            # This hack just ensures that it is the weakest softmax activation, since softmax can't
            # take inf as input
            alpha[1] = torch.min(alpha) - 0.001
            alpha_softmax = F.softmax(alpha)

            return torch.max(alpha_softmax)

        k = 2
        if len(in_edges) >= k:
            if any(e.has("alpha") or (e.has("final") and e.final) for _, e in in_edges):
                # We are in the one-shot case
                for _, data in in_edges:
                    if data.has("final") and data.final:
                        return  # We are looking at an out node
                    data.alpha.data[1] = -float("Inf")
                sorted_edge_ids = sorted(in_edges, key=_largest_post_softmax_weight, reverse=True)
                keep_edges, _ = zip(*sorted_edge_ids[:k])
                for edge_id, edge_data in in_edges:
                    if edge_id not in keep_edges:
                        edge_data.delete()
            else:
                # We are in the discrete case (e.g. random search)
                for _, data in in_edges:
                    if isinstance(data.op, list) and data.op[1].get_op_name == "Zero":
                        data.op.pop(1)
                if any(e.has("final") and e.final for _, e in in_edges):
                    return  # TODO: how about mixed final and non-final?
                else:
                    for _ in range(len(in_edges) - k):
                        in_edges[random.randint(0, len(in_edges) - 1)][1].delete()

    def get_compact(self):
        if self.compact is None:
            self.compact = convert_naslib_to_compact(self)
        return self.compact

    def get_hash(self):
        return self.get_compact()

    def set_compact(self, compact):
        # This will update the edges in the naslib object to match compact
        self.compact = compact

        if self.instantiate_model == True:
            convert_compact_to_naslib(compact, self)

    def set_spec(self, compact, dataset_api=None):
        self.set_compact(make_compact_immutable(compact))

    def sample_random_labeled_architecture(self):
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api=None, load_labeled=False, no_zero=True):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        compact = [[], []]
        for i in range(self.num_vertices):
            ops = np.random.choice(range(self.num_ops-1), self.num_vertices)

            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            compact[0].extend(
                [(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])]
            )
            compact[1].extend(
                [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
            )

        self.set_compact(compact)

    @staticmethod
    def get_configspace(
        path_to_configspace_obj=os.path.join(
            get_project_root(), "search_spaces/nasbench301/configspace.json"
        )
    ):
        """
        Returns the ConfigSpace object for the search space

        Args:
            path_to_configspace_obj: path to ConfigSpace json encoding

        Returns:
            ConfigSpace.ConfigutationSpace: a ConfigSpace object
        """
        with open(path_to_configspace_obj, "r") as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)
        return config_space

    def get_type(self):
        return "nasbench301"

    def get_loss_fn(self):
        return F.cross_entropy

    def forward_before_global_avg_pool(self, x):
        outputs = []
        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x)

        assert len(outputs) == 1
        return outputs[0]

    def clear_graph_tensors(self):
        def _clear_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = _clear_tensors(value)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = _clear_tensors(obj[i])
            elif hasattr(obj, '__dict__'):
                for key in vars(obj):
                    setattr(obj, key, _clear_tensors(getattr(obj, key)))
            return obj

        def _clear_graph(graph):
            for node_idx in graph.nodes:
                node = graph.nodes[node_idx]
                if "subgraph" in node:
                    _clear_graph(node["subgraph"])
                if "input" in node:
                    node["input"] = _clear_tensors(node["input"])
                if "comb_op" in node and isinstance(node["comb_op"], AbstractCombOp):
                    for u, v in graph.in_edges(node_idx):
                        edge_data = graph.get_edge_data(u, v)
                        edge_data.op = _clear_tensors(edge_data.op)

                for neighbor_idx in graph.neighbors(node_idx):
                    edge_data = graph.get_edge_data(node_idx, neighbor_idx)
                    if isinstance(edge_data.op, Graph):
                        _clear_graph(edge_data.op)
                    else:
                        edge_data.op = _clear_tensors(edge_data.op)

            # Ensure that any graph-level tensors are also detached
            for attr in vars(graph):
                setattr(graph, attr, _clear_tensors(getattr(graph, attr)))

        _clear_graph(self)


def channel_concat(tensors):
    return torch.cat(tensors, dim=1)


def channel_maps(reduction_cell_indices, max_index):
    # calculate the mapping from edge indices to the respective channel

    assert len(reduction_cell_indices) == 2
    r_1, r_2 = reduction_cell_indices
    channel_map_from = {}
    channel_map_from.update({i: 0 for i in range(2, r_1)})
    channel_map_from.update({i: 1 for i in range(r_1, r_2)})
    channel_map_from.update({i: 2 for i in range(r_2, max_index)})

    channel_map_to = {}
    channel_map_to.update({i: 0 for i in range(3, r_1 + 1)})
    channel_map_to.update({i: 1 for i in range(r_1 + 1, r_2 + 1)})
    channel_map_to.update({i: 2 for i in range(r_2 + 1, max_index)})

    return channel_map_from, channel_map_to
