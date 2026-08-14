import json
import os
import fcntl
from .utils import get_project_root

"""
This file loads any dataset files or api's needed by the Trainer or ZeroCostPredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_nasbench301_api(dataset):
    if dataset != 'cifar10':
        return None
    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/automl/nasbench301@no_gin')

    # Paths to v1.0 model files and data file.
    download_path = os.path.join(get_project_root(), "data")
    configured_models_path = os.environ.get("NB301_MODELS_ROOT")
    nb_models_path = configured_models_path or os.path.join(download_path, "nb_models_1.0")
    os.makedirs(download_path, exist_ok=True)

    nb301_model_path=os.path.join(nb_models_path, "xgb_v1.0")
    nb301_runtime_path=os.path.join(nb_models_path, "lgb_runtime_v1.0")

    required_models = [nb301_model_path, nb301_runtime_path]
    if configured_models_path:
        if not all(os.path.exists(model) for model in required_models):
            raise FileNotFoundError(
                "NB301_MODELS_ROOT must contain xgb_v1.0 and lgb_runtime_v1.0: "
                + configured_models_path
            )
    else:
        lock_path = os.path.join(download_path, ".nb301_models_1.0.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not all(os.path.exists(model) for model in required_models):
                nasbench301.download_models(version='1.0', delete_zip=True,
                                            download_dir=download_path)

    models_not_found_msg = "Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510"

    # Verify the model and data files exist
    assert os.path.exists(nb_models_path), f"Could not find {nb_models_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_model_path), f"Could not find {nb301_model_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_runtime_path), f"Could not find {nb301_runtime_path}. {models_not_found_msg}"

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    nb301_model = [performance_model, runtime_model]

    return {
        "nb301_model": nb301_model,
    }


def get_dataset_api(search_space=None, dataset=None):
    if search_space == "nasbench301":
        return get_nasbench301_api(dataset=dataset)
    raise NotImplementedError()


def get_zc_benchmark_api(search_space, dataset):

    datafile_path = os.path.join(get_project_root(), "data", f"zc_{search_space}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data[dataset]


def load_sampled_architectures(search_space, postfix=''):
    datafile_path = os.path.join(get_project_root(), "data", "archs", f"archs_{search_space}{postfix}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data
