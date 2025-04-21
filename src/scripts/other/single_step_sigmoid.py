# python src/scripts/single_step.py -k v2_main -c src/scripts/paper/configs/final.json -o src/scripts/paper/figures/single_step_main.pdf --moving_avg 5

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scaling.fitting_functions import get_coefficients, log_sigmoid, sigmoid
from scaling.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
FONTSIZE = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="Path to write output figure"
    )
    args = parser.parse_args()

    args.keys = get_task_sets(args.keys)

    return args


def fit_single_step(data_by_name, task_name, _max=None, _min=None, use_log_sigmoid=False):
    train_xs, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_xs += data["fs"]
            train_ys += data["xs"]

    if use_log_sigmoid:
        train_xs = np.log(train_xs)

    if _max is None:
        _max = tasks[task_name].task_maximum
    if _min is None:
        _min = tasks[task_name].task_minimum

    # add ideal points (these are not plotted)
    if not use_log_sigmoid:
        train_xs.append(0.0)
        train_ys.append(_max)
        # train_xs.append(max(train_xs))
        # train_ys.append(tasks[task_name].task_minimum)

    # fit the parameters
    if use_log_sigmoid:
        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            log_sigmoid,
            p0=[0.2, 21, 0.5],
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            disp=False,
            return_cov=True,
        )
    else:
        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            sigmoid,
            p0=[0.5, 2, 10, 0],  # a, x0, k, b
            bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
            # bounds=([tasks[task_name].task_minimum - 1.0, 0.0, 0.0, tasks[task_name].task_maximum - 0.0001], [tasks[task_name].task_minimum - 0.9999, np.inf, np.inf, tasks[task_name].task_maximum]),
            disp=False,
            return_cov=True,
        )

    return coefficients


def predict_single_step(data_by_name, coefficients, use_log_sigmoid=False):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    predict_fn = log_sigmoid if use_log_sigmoid else sigmoid
    # fit_fn = log_sigmoid_fit if use_log_sigmoid else sigmoid_fit
    # grad_fit_fn = grad_log_sigmoid_fit if use_log_sigmoid else grad_sigmoid_fit

    dmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
    dmax = 1.5 * max([max(data["fs"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "fs": data["fs"],
            "ys": [predict_fn(np.log(x), *coefficients) for x in data["fs"]],  # type: ignore
        }
        fs = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        plotted_predicted_data_by_name[name] = {
            "fs": fs,
            "ys": [predict_fn(np.log(x), *coefficients) for x in fs],  # type: ignore
        }

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["fs"], data["xs"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_sigmoid(coefficients, use_log_sigmoid=False):
    if use_log_sigmoid:
        a, x0, k = coefficients
        return f"Acc(L) = 1 - {-a:.2f} * log(1 - 1/(1 + e^(-{k:.2f}(L - {x0:.2f})))"
    else:
        a, x0, k, b = coefficients
        return f"Acc(L) = {a:.2f} / (1 + \\exp (-{k:.2f}(L - {x0:.2f}))) + {b:.2f}"


def plot_single_step(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data_by_name,
    task_name,
    fit_str,
    ax=plt.gca(),
):
    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["fs"],
            data["ys"],
            color=config.color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"{config.label} (fitted)" if config.mode == "train" else None,
        )

    # plot the actual and predicted data
    unsigned_rel_errors = []
    num_eval_annotation = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        for i, (d, y, ln) in enumerate(zip(data["fs"], data["xs"], data["ls"])):
            ax.scatter(
                d,
                y,
                color=config.color,
                marker=MARKERS[ln] if config.mode == "train" else "x",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        for d, y, y_pred in zip(data["fs"], data["xs"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.scatter(
                    d,
                    y_pred,
                    color=config.color,
                    marker="o",
                    s=20,
                    label=f"{config.label} (predicted)",
                )
                ax.annotate(
                    f"{abs(rel_error * 100):.1f}%",
                    (d, y_pred),
                    textcoords="offset points",
                    xytext=(10, -5 + 10 * num_eval_annotation),
                    ha="left",
                    va="bottom",
                    fontsize=FONTSIZE,
                    color=config.color,
                )
                num_eval_annotation += 1
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("FLOPs (F)", fontsize=FONTSIZE)
    ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    ax.set_title(
        f"{tasks[task_name].display_name} ({avg_unsigned_rel_error * 100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False
    )

    use_log_sigmoid = True

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg
        )

        for name, data in data_by_name.items():
            data["ys"] = data["xs"]  # to deal with refactor

        # fit the parameters
        coefficients = fit_single_step(data_by_name, task_name, use_log_sigmoid=use_log_sigmoid)

        # make predictions
        (
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            (y, y_pred, rel_error),
        ) = predict_single_step(data_by_name, coefficients, use_log_sigmoid=use_log_sigmoid)
        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"

        plot_single_step(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_sigmoid(coefficients, use_log_sigmoid=use_log_sigmoid),
            axes[i // num_cols][i % num_cols],
        )

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    # delete x-axis labels for all but the bottom row
    for i in range(num_cols):
        for j in range(num_rows):
            if j != num_rows - 1:
                axes[j][i].set_xlabel("")
            if i != 0:
                axes[j][i].set_ylabel("")

            axes[j][i].legend().remove()

    fig.tight_layout(w_pad=0.01)
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1.07),
        handletextpad=0.3,
        columnspacing=0.7,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")

    print(results)


if __name__ == "__main__":
    main()
