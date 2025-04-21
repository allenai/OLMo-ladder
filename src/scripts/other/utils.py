import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from scaling.task_utils import KEYS_BY_KEY, WEIGHT_BY_KEY


@dataclass
class ExtrapolateNConfig:
    path: str
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

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """


def get_data_by_name(
    configs: Dict[str, ExtrapolateNConfig], keys: List[str], min_step: Optional[int] = None
):
    data_by_name: Dict = defaultdict(
        lambda: {"ns": [], "ds": [], "hs": [], "s1s": [], "s2s": [], "ys": []}
    )
    for name, config in configs.items():
        n = config.n
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            lam = 0.999
            s1 = 0.0
            s2 = 0.0
            s2_momentum = 0
            last_lr = 0.0
            last_fake_lr = 0.0
            last_d = 0
            encountered_ds = set()
            for row in reader:
                d = int(float(row["throughput/total_tokens"]))
                if d in encountered_ds:
                    continue
                batch_size = int(row["batch_size_in_tokens"])
                steps = (d - last_d) / batch_size
                lr = float(row["optim/learning_rate_group0"])
                if lr > last_lr:  # warmup phase
                    fake_lr = float(row["learning_rate_peak"])
                    last_fake_lr = float(row["learning_rate_peak"])
                else:  # anneal phase
                    fake_lr = lr
                h = lr / float(row["learning_rate_peak"])
                s1 += fake_lr * steps
                s2_momentum = lam**steps * s2_momentum + (last_fake_lr - fake_lr) * steps
                s2 += s2_momentum
                last_lr = lr
                last_fake_lr = fake_lr
                last_d = d
                encountered_ds.add(d)
                y = np.average(
                    [float(row[key]) for key in keys],
                    weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in keys],
                )
                if min_step is not None and d < min_step * batch_size:
                    continue
                data_by_name[name]["ns"].append(n)
                data_by_name[name]["ds"].append(d)
                data_by_name[name]["hs"].append(h)
                data_by_name[name]["s1s"].append(s1)
                data_by_name[name]["s2s"].append(s2)
                data_by_name[name]["ys"].append(y)
    return data_by_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default="",
        help="For avg metrics. Use one of [all-val-lm, all-bpb]",
    )
    parser.add_argument(
        "--num_to_avg",
        type=int,
        default=1,
        help="Number of final ckpts to average (for final loss fitting)",
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="Path to write output figure"
    )
    args = parser.parse_args()

    args.keys = KEYS_BY_KEY[args.key]

    return args
