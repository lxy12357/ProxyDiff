from .nasbench301.graph import NasBench301SearchSpace

supported_search_spaces = {
    "nasbench301": NasBench301SearchSpace,
}

dataset_n_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet16-120": 120,
    "svhn": 10,
    "ninapro": 18,
    "scifar100": 100,
}

dataset_to_channels = {
    "cifar10": 3,
    "cifar100": 3,
    "imagenet16-120": 3,
    "svhn": 3,
    "ninapro": 1,
    "scifar100": 3,
}

def get_search_space(name, dataset,metric_names=[],metric_epoch_accumulate=[]):
    search_space_cls = supported_search_spaces[name.lower()]

    try:
        in_channels = dataset_to_channels[dataset.lower()]
    except KeyError:
        in_channels = 3

    try:
        n_classes = dataset_n_classes[dataset.lower()]
    except KeyError:
        n_classes = -1

    if name == 'nasbench301':
        auxiliary = True if dataset.lower() == "cifar10" else False
        return search_space_cls(n_classes=n_classes, in_channels=in_channels,
                                auxiliary=auxiliary,metric_names=metric_names,metric_epoch_accumulate=metric_epoch_accumulate)
    raise NotImplementedError(f'{name} search space not implemented')
