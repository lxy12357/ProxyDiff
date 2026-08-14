"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip
based on https://github.com/mohsaied/zero-cost-nas
"""
import torch
import logging
import math

from .predictor import Predictor
from .pruners import predictive

logger = logging.getLogger(__name__)


class ZeroCost(Predictor):
    def __init__(self, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type
        self.dataload = "random"
        # single batch
        self.num_imgs_or_batches = 1
        # multiple batch
        # self.num_imgs_or_batches = "all"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def query(self, graph, dataloader=None, info=None,epoch_num=1,epoch_by_epoch=False,data=None):
        loss_fn = graph.get_loss_fn()

        n_classes = graph.num_classes

        if isinstance(self.method_type, list):
            measure_names = self.method_type
        else:
            measure_names = [self.method_type]

        score = predictive.find_measures(
                net_orig=graph,
                dataloader=dataloader,
                data=data,
                dataload_info=(self.dataload, self.num_imgs_or_batches, n_classes),
                device=self.device,
                loss_fn=loss_fn,
                measure_names=measure_names,
                epoch_num=epoch_num,
                epoch_by_epoch = epoch_by_epoch
            )

        if not isinstance(score, list) and (math.isnan(score) or math.isinf(score)):
            score = -1e8

        if self.method_type == 'synflow':
            if score == 0.:
                return score

            score = math.log(score) if score > 0 else -math.log(-score)

        return score
