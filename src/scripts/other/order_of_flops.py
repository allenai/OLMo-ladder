import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PARAMS = {
    "190M": 190354176,
    "370M": 371262464,
    "600M": 597382464,
    "760M": 758220288,
    "1B": 1279395840,
    "3B": 3169537280,
    "7B": 6887575552,
    "13B": 13202396160,
}

MODEL_COLORS = {"190M": "darkred", "370M": "darkorange", "760M": "darkgreen", "1B": "teal"}

sorted_flops = [
    ("190M-1xC", 7246369391459696640),
    ("190M-2xC", 14492738782919393280),
    ("370M-1xC", 25571986360311480320),
    ("190M-5xC", 36231846957298483200),
    ("370M-2xC", 51143972720622960640),
    ("190M-10xC", 72463693914596966400),
    ("760M-1xC", 96655556181680455680),
    ("370M-5xC", 127859931801557401600),
    ("760M-2xC", 193311112363360911360),
    ("370M-10xC", 255719863603114803200),
    ("1B-1xC", 258670076884942848000),
    ("760M-5xC", 483277780908402278400),
    ("1B-2xC", 517340153769885696000),
    ("760M-10xC", 966555561816804556800),
    ("1B-5xC", 1293350384424714240000),
    ("1B-10xC", 2586700768849428480000),
]


def plot_flops(output_path):
    markers = {"1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(3, 2.5))

    eps = {"1xC": 0, "2xC": 3, "5xC": 7, "10xC": 12}

    for label, flops in sorted_flops:
        model_size, multiplier = label.split("-")
        params = MODEL_PARAMS[model_size]
        ax.scatter(flops, params, color=MODEL_COLORS[model_size], marker=markers[multiplier])
        ax.annotate(
            label,
            (flops, params),
            textcoords="offset points",
            xytext=(eps[multiplier], 5),
            ha="center",
            fontsize=4,
            color="brown",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Flops")
    ax.set_ylabel("Model size")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_flops("sorted_flops.pdf")
