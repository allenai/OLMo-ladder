import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from scaling.task_utils import *


@dataclass
class FinalConfig:
    paths: List[str]
    """
    Path containing the W&B downloaded data and metadata.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """

    n: int
    """
    The model size (non-embedding parameter count).
    """

    flops: int
    """
    The number of FLOPs for the model size.
    """

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """

    use_last_n_percentage: int = 100
    """
    The percent of data points used. Defaults to 100%.
    """


MODEL_FLOPS = {
    "190M": 1903391232,
    "370M": 3443922944,
    "600M": 5180751744,
    "760M": 6373843968,
    "1.3B": 10109071360,
    "3.2B": 22970355200,
    "7B": 49412071424,
    "13B": 91335915520,
}

MODEL_PARAMS = {
    "190M": 190354176,
    "370M": 371262464,
    "600M": 597382464,
    "760M": 758220288,
    "1.3B": 1279395840,
    "3.2B": 3169537280,
    "7B": 6887575552,
    "13B": 13202396160,
}


def get_final_configs(config_path: str):
    with open(config_path) as f:
        configs = json.load(f)
        configs = {name: FinalConfig(**config) for name, config in configs.items()}
    return configs


def moving_average(arr, n):
    ret = np.cumsum(arr, dtype=float)
    if len(ret) < n:
        return ret / np.arange(1, len(ret) + 1)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate([ret[: n - 1] / np.arange(1, n), ret[n - 1 :] / n])


def get_length(path):
    try:
        return path.split("/")[-1].split(".csv")[0].split("-")[1]
    except IndexError:
        return ""


def get_step1_data_by_name(configs, task_name, y_metric="rc_bpb", moving_avg=1):
    task = tasks[task_name]
    if y_metric == "rc_bpb":
        keys = task.get_loss_keys()
    elif y_metric == "rc_acc":
        keys = task.get_accuracy_keys()
    elif y_metric == "c4":
        keys = ["eval/c4_en-validation/CrossEntropyLoss"]
    elif y_metric == "rc_soft_log":
        keys = task.get_accuracy_keys()
        keys = [
            key.replace("/downstream/", "/downstream_soft_log/") if "_len_norm" in key else key
            for key in keys
        ]
        keys = [key.replace("_len_norm", "_soft_log") for key in keys]
    else:
        raise ValueError(f"Invalid y_metric: {y_metric}")

    data_by_name: Dict = defaultdict(lambda: {"ns": [], "ds": [], "xs": [], "ls": [], "fs": []})
    for name, config in configs.items():
        n = config.n
        for path in config.paths:
            length = get_length(path)
            with open(path) as file_ref:
                reader = csv.DictReader(file_ref)
                rows = [row for row in reader]
                rows = rows[-moving_avg:]
                ds, xs, fs = [], [], []
                for row in rows:
                    if "throughput/total_tokens" in row:
                        d = int(float(row["throughput/total_tokens"]))
                    else:
                        d = int(float(row["_step"])) * int(float(row["batch_size_in_tokens"]))
                    f = float(d * MODEL_FLOPS[name.split("-")[0]])
                    x = np.average(
                        [float(row[key]) for key in keys],
                        weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in keys],
                    )
                    if y_metric == "rc_soft_log":
                        x *= -1
                    ds.append(d)
                    xs.append(x)
                    fs.append(f)
                d = ds[-1]
                x = np.mean(xs)
                f = fs[-1]
                data_by_name[name]["ns"].append(n)
                data_by_name[name]["ds"].append(d)
                data_by_name[name]["xs"].append(x)
                data_by_name[name]["ls"].append(length)
                data_by_name[name]["fs"].append(f)
        data_by_name[name]["mode"] = config.mode
    return data_by_name


def get_step2_data_by_name(
    configs,
    task_name,
    x_metric="rc_bpb",
    y_metric="rc_acc",
    moving_avg=1,
    skip_perc=0.0,
    last_n_points=-1,
):
    task = tasks[task_name]
    if x_metric == "rc_bpb":
        loss_keys = task.get_loss_keys()
    elif x_metric == "rc_soft_log":
        loss_keys = task.get_accuracy_keys()
        loss_keys = [
            key.replace("/downstream/", "/downstream_soft_log/") if "_len_norm" in key else key
            for key in loss_keys
        ]
        loss_keys = [key.replace("_len_norm", "_soft_log") for key in loss_keys]
    elif x_metric == "c4":
        loss_keys = ["eval/c4_en-validation/CrossEntropyLoss"]
    else:
        raise ValueError(f"Invalid x_metric: {x_metric}")
    if y_metric == "rc_acc":
        accuracy_keys = task.get_accuracy_keys()
    elif y_metric == "mc_acc":
        accuracy_keys = task.get_mc_accuracy_keys()
    else:
        raise ValueError(f"Invalid y_metric: {y_metric}")

    data_by_name: Dict = defaultdict(lambda: {"xs": [], "ys": [], "ds": [], "ns": [], "ls": []})

    for name, config in configs.items():
        if name == "external":
            xs, ys = [], []
            for path in config.paths:
                with open(path) as f:
                    data = json.load(f)
                    x = np.average(
                        [float(data[key]) for key in loss_keys],
                        weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in loss_keys],
                    )
                    y = np.average(
                        [float(data[key]) for key in accuracy_keys],
                        weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in accuracy_keys],
                    )
                    xs.append(x)
                    ys.append(y)
            data_by_name[name] = {"xs": xs, "ys": ys, "ds": [], "ns": [], "ls": []}

        else:
            n = config.n
            for path in config.paths:
                length = get_length(path)
                with open(path) as file_ref:
                    reader = csv.DictReader(file_ref)
                    rows = [row for row in reader]
                    xs, ys, ds, ns, ls = [], [], [], [], []
                    for row in rows:
                        if "throughput/total_tokens" in row:
                            d = int(float(row["throughput/total_tokens"]))
                        else:
                            d = int(float(row["_step"])) * int(float(row["batch_size_in_tokens"]))
                        x = np.average(
                            [float(row[key]) for key in loss_keys],
                            weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in loss_keys],
                        )
                        if x_metric == "rc_soft_log":
                            x *= -1

                        y = np.average(
                            [float(row[key]) for key in accuracy_keys],
                            weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in accuracy_keys],
                        )
                        if y_metric == "rc_soft_log":
                            y *= -1

                        xs.append(x)
                        ys.append(y)
                        ds.append(d)
                        ns.append(n)
                        ls.append(length)

                    if config.mode == "train":
                        # skip initial ckpts

                        if skip_perc == 1:
                            xs = [xs[-1]]
                            ys = [ys[-1]]
                            ds = [ds[-1]]
                            ns = [ns[-1]]
                            ls = [ls[-1]]
                        else:
                            xs = xs[int(np.ceil(skip_perc * len(xs))) :]
                            ys = ys[int(np.ceil(skip_perc * len(ys))) :]
                            ds = ds[int(np.ceil(skip_perc * len(ds))) :]
                            ns = ns[int(np.ceil(skip_perc * len(ns))) :]
                            ls = ls[int(np.ceil(skip_perc * len(ls))) :]

                    # apply moving_avg
                    xs = moving_average(xs, n=moving_avg).tolist()
                    ys = moving_average(ys, n=moving_avg).tolist()
                    # ys = ys[moving_avg-1:]
                    # ds = ds[moving_avg-1:]
                    # ns = ns[moving_avg-1:]
                    # ls = ls[moving_avg-1:]

                    if config.mode == "train":
                        # last n points
                        if last_n_points > 0:
                            xs = xs[-last_n_points:]  # type: ignore
                            ys = ys[-last_n_points:]  # type: ignore
                            ds = ds[-last_n_points:]
                            ns = ns[-last_n_points:]
                            ls = ls[-last_n_points:]

                    data_by_name[name]["xs"] += xs
                    data_by_name[name]["ys"] += ys
                    data_by_name[name]["ds"] += ds
                    data_by_name[name]["ns"] += ns
                    data_by_name[name]["ls"] += ls

        data_by_name[name]["mode"] = config.mode

    return data_by_name


## Printing and plotting utility functions

MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*", "": "o"}


def prettify(rel_error, is_percentage=True):
    if is_percentage:
        return f"{rel_error * 100:+.1f}%"
    else:
        return f"{rel_error:.2f}"


def get_ax(name):
    if "1xC" in name:
        return 0
    if "2xC" in name:
        return 1
    if "5xC" in name:
        return 2
    if "10xC" in name:
        return 3
    return 4
