from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class DownstreamTaskPrediction:
    task_loss_key: Union[str, List[str]]
    task_accuracy_key: Union[str, List[str]]
    task_mc_loss_key: Union[str, List[str]]
    task_mc_accuracy_key: Union[str, List[str]]
    display_name: str
    task_soft_loss_key: Union[str, List[str]] = ""
    task_log_soft_loss_key: Union[str, List[str]] = ""
    task_minimum: float = 0.25
    task_maximum: float = 1.0

    def get_loss_keys(self):
        return self.task_loss_key if isinstance(self.task_loss_key, list) else [self.task_loss_key]

    def get_accuracy_keys(self):
        return (
            self.task_accuracy_key
            if isinstance(self.task_accuracy_key, list)
            else [self.task_accuracy_key]
        )

    def get_mc_loss_keys(self):
        return (
            self.task_mc_loss_key
            if isinstance(self.task_mc_loss_key, list)
            else [self.task_mc_loss_key]
        )

    def get_mc_accuracy_keys(self):
        return (
            self.task_mc_accuracy_key
            if isinstance(self.task_mc_accuracy_key, list)
            else [self.task_mc_accuracy_key]
        )

    def get_soft_loss_keys(self):
        return (
            self.task_soft_loss_key
            if isinstance(self.task_soft_loss_key, list)
            else [self.task_soft_loss_key]
        )

    def get_log_soft_loss_keys(self):
        return (
            self.task_log_soft_loss_key
            if isinstance(self.task_log_soft_loss_key, list)
            else [self.task_log_soft_loss_key]
        )


minimums_rc: Dict[str, float] = {
    "piqa": 0.5,
    "socialiqa": 1 / 3,
    "csqa": 0.2,
    "boolq": 0.5,
    "winogrande": 0.5,
}
maximums_rc: Dict[str, float] = {}  # {"mmlu_stem": 0.9, "arc_easy": 0.85}


v2_minimums_rc: Dict[str, float] = {
    "piqa_val": 0.5,
    "socialiqa_val": 1 / 3,
    "csqa_val": 0.2,
    "boolq_val": 0.5,
    "winogrande_val": 0.5,
}
v2_maximums_rc: Dict[str, float] = {
    # "mmlu_avg_test": 1.06,
    # "arc_challenge_test": 1.65,
    # "arc_easy_test": 1.40,
    # "piqa_val": 1.53,
    # "csqa_val": 1.10,
    # "socialiqa_val": 0.73,
    # "openbookqa_test": 1.94,
}

validation = [
    "c4_en-validation",
    "dolma_books-validation",
    "dolma_common-crawl-validation",
    "dolma_pes2o-validation",
    "dolma_reddit-validation",
    "dolma_stack-validation",
    "dolma_wiki-validation",
    "ice-validation",
    "m2d2_s2orc-validation",
    "pile-validation",
    "wikitext_103-validation",
]

v3_validation = [
    "v3-small-c4_en-validation",
    "v3-small-dolma_books-validation",
    "v3-small-dolma_common-crawl-validation",
    "v3-small-dolma_pes2o-validation",
    "v3-small-dolma_reddit-validation",
    "v3-small-dolma_stack-validation",
    "v3-small-dolma_wiki-validation",
    "v3-small-ice-validation",
    "v3-small-m2d2_s2orc-validation",
    #'v3-small-pile-validation',
    "v3-small-wikitext_103-validation",
]

core_names = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
]
core_small_names = [
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "piqa",
    "csqa",
    "socialiqa",
    "openbookqa",
]
mmlu_names = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]

display_names = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "csqa": "CommonsenseQA",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "socialiqa": "Social IQa",
    "winogrande": "Winogrande",
}

core_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb_bpb",
        task_accuracy_key=(
            f"eval/downstream/{key}_rc_5shot_len_norm"
            if key not in ["arc_easy", "winogrande", "boolq"]
            else f"eval/downstream/{key}_rc_5shot_acc"
        ),
        task_mc_loss_key=f"eval/downstream_bpb/{key}_mc_5shot_bpb_bpb",
        task_mc_accuracy_key=f"eval/downstream/{key}_mc_5shot_acc",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 1.0),
        display_name=display_names.get(key, key),
    )
    for key in core_names
}
core_small_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb_bpb",
        task_accuracy_key=(
            f"eval/downstream/{key}_rc_5shot_len_norm"
            if key not in ["arc_easy", "winogrande", "boolq"]
            else f"eval/downstream/{key}_rc_5shot_acc"
        ),
        task_mc_loss_key=f"eval/downstream_bpb/{key}_mc_5shot_bpb_bpb",
        task_mc_accuracy_key=f"eval/downstream/{key}_mc_5shot_acc",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 1.0),
        display_name=display_names.get(key, key),
    )
    for key in core_small_names
}
core_small_avg_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    "core_small_avg_5shot": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_rc_5shot_bpb_bpb" for key in core_small_names],
        task_accuracy_key=[
            (
                f"eval/downstream/{key}_rc_5shot_len_norm"
                if key not in ["arc_easy", "winogrande", "boolq"]
                else f"eval/downstream/{key}_rc_5shot_acc"
            )
            for key in core_small_names
        ],
        task_mc_loss_key=[
            f"eval/downstream_bpb/{key}_mc_5shot_bpb_bpb" for key in core_small_names
        ],
        task_mc_accuracy_key=[f"eval/downstream/{key}_mc_5shot_acc" for key in core_small_names],
        task_minimum=0.25,
        task_maximum=1.0,
        display_name="core_small_avg",
    )
}
mmlu_var_tasks: Dict[str, DownstreamTaskPrediction] = {
    "mmlu_avg_var": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_var_bpb_bpb" for key in mmlu_names],
        task_accuracy_key=[f"eval/downstream/{key}_var_len_norm" for key in mmlu_names],
        task_mc_loss_key=[f"eval/downstream_bpb/{key}_mc_5shot_bpb_bpb" for key in mmlu_names],
        task_mc_accuracy_key=[f"eval/downstream/{key}_mc_5shot_len_norm" for key in mmlu_names],
        task_minimum=0.25,
        task_maximum=1.0,  # 0.9,
        display_name="MMLU",
    )
}
mmlu_subset_var_tasks: Dict[str, DownstreamTaskPrediction] = {
    key: DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_var_bpb_bpb",
        task_accuracy_key=f"eval/downstream/{key}_var_len_norm",
        task_mc_loss_key=f"eval/downstream_bpb/{key}_mc_5shot_bpb_bpb",
        task_mc_accuracy_key=f"eval/downstream/{key}_mc_5shot_len_norm",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 0.9),
        display_name="MMLU",
    )
    for key in mmlu_names
}


v2_core_names = [
    "hellaswag_val",
    "arc_easy_val",
    "arc_easy_test",
    "arc_challenge_val",
    "arc_challenge_test",
    "boolq_val",
    "csqa_val",
    "openbookqa_val",
    "openbookqa_test",
    "piqa_val",
    "socialiqa_val",
    "winogrande_val",
]
v2_core_small_names = [
    "hellaswag_val",
    "arc_challenge_test",
    "arc_easy_test",
    "piqa_val",
    "csqa_val",
    "socialiqa_val",
    "openbookqa_test",
]
v2_mmlu_val_names = [
    "mmlu_stem_val",
    "mmlu_humanities_val",
    "mmlu_social_sciences_val",
    "mmlu_other_val",
]
v2_mmlu_test_names = [
    "mmlu_stem_test",
    "mmlu_humanities_test",
    "mmlu_social_sciences_test",
    "mmlu_other_test",
]

v2_core_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb",
        task_soft_loss_key=f"eval/downstream_soft/{key}_rc_5shot_soft",
        task_log_soft_loss_key=f"eval/downstream_soft_log/{key}_rc_5shot_soft_log",
        task_accuracy_key=(
            f"eval/downstream/{key}_rc_5shot_len_norm"
            if "boolq" not in key
            else f"eval/downstream/{key}_rc_5shot_acc"
        ),
        task_mc_loss_key=f"eval/downstream_bpb/{key}_mc_5shot_bpb",
        task_mc_accuracy_key=f"eval/downstream/{key}_mc_5shot_acc",
        task_minimum=v2_minimums_rc.get(key, 0.25),
        task_maximum=v2_maximums_rc.get(key, 1.0),
        display_name=display_names.get(key.removesuffix("_val").removesuffix("_test"), key),
    )
    for key in v2_core_names
}
v2_core_small_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb",
        task_soft_loss_key=f"eval/downstream_soft/{key}_rc_5shot_soft",
        task_log_soft_loss_key=f"eval/downstream_soft_log/{key}_rc_5shot_soft_log",
        task_accuracy_key=(
            f"eval/downstream/{key}_rc_5shot_len_norm"
            if "boolq" not in key
            else f"eval/downstream/{key}_rc_5shot_acc"
        ),
        task_mc_loss_key=f"eval/downstream_bpb/{key}_mc_5shot_bpb",
        task_mc_accuracy_key=f"eval/downstream/{key}_mc_5shot_acc",
        task_minimum=v2_minimums_rc.get(key, 0.25),
        task_maximum=v2_maximums_rc.get(key, 1.0),
        display_name=display_names.get(key.removesuffix("_val").removesuffix("_test"), key),
    )
    for key in v2_core_small_names
}

v2_mmlu_avg_val_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    "mmlu_avg_val_5shot": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_rc_5shot_bpb" for key in v2_mmlu_val_names],
        task_soft_loss_key=[
            f"eval/downstream_soft/{key}_rc_5shot_soft" for key in v2_mmlu_val_names
        ],
        task_log_soft_loss_key=[
            f"eval/downstream_soft_log/{key}_rc_5shot_soft_log" for key in v2_mmlu_val_names
        ],
        task_accuracy_key=[f"eval/downstream/{key}_rc_5shot_len_norm" for key in v2_mmlu_val_names],
        task_mc_loss_key=[f"eval/downstream_bpb/{key}_mc_5shot_bpb" for key in v2_mmlu_val_names],
        task_mc_accuracy_key=[
            f"eval/downstream/{key}_mc_5shot_len_norm" for key in v2_mmlu_val_names
        ],
        task_minimum=v2_minimums_rc.get("mmlu_avg_val", 0.25),
        task_maximum=v2_maximums_rc.get("mmlu_avg_val", 1.0),
        display_name="MMLU",
    )
}

v2_mmlu_avg_test_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    "mmlu_avg_test_5shot": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_rc_5shot_bpb" for key in v2_mmlu_test_names],
        task_soft_loss_key=[
            f"eval/downstream_soft/{key}_rc_5shot_soft" for key in v2_mmlu_test_names
        ],
        task_log_soft_loss_key=[
            f"eval/downstream_soft_log/{key}_rc_5shot_soft_log" for key in v2_mmlu_test_names
        ],
        task_accuracy_key=[
            f"eval/downstream/{key}_rc_5shot_len_norm" for key in v2_mmlu_test_names
        ],
        task_mc_loss_key=[f"eval/downstream_bpb/{key}_mc_5shot_bpb" for key in v2_mmlu_test_names],
        task_mc_accuracy_key=[
            f"eval/downstream/{key}_mc_5shot_len_norm" for key in v2_mmlu_test_names
        ],
        task_minimum=v2_minimums_rc.get("mmlu_avg_test", 0.25),
        task_maximum=v2_maximums_rc.get("mmlu_avg_test", 1.0),
        display_name="MMLU",
    )
}

v2_main_avg_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    "main_avg_5shot": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_rc_5shot_bpb" for key in v2_core_small_names]
        + [f"eval/downstream_bpb/{key}_rc_5shot_bpb" for key in v2_mmlu_test_names],
        task_soft_loss_key=[
            f"eval/downstream_soft/{key}_rc_5shot_soft" for key in v2_core_small_names
        ]
        + [f"eval/downstream_soft/{key}_rc_5shot_soft" for key in v2_mmlu_test_names],
        task_log_soft_loss_key=[
            f"eval/downstream_soft_log/{key}_rc_5shot_soft_log" for key in v2_core_small_names
        ]
        + [f"eval/downstream_soft_log/{key}_rc_5shot_soft_log" for key in v2_mmlu_test_names],
        task_accuracy_key=[
            (
                f"eval/downstream/{key}_rc_5shot_len_norm"
                if "boolq" not in key
                else f"eval/downstream/{key}_rc_5shot_acc"
            )
            for key in v2_core_small_names
        ]
        + [f"eval/downstream/{key}_rc_5shot_len_norm" for key in v2_mmlu_test_names],
        task_mc_loss_key=[f"eval/downstream_bpb/{key}_mc_5shot_bpb" for key in v2_core_small_names]
        + [f"eval/downstream_bpb/{key}_mc_5shot_bpb" for key in v2_mmlu_test_names],
        task_mc_accuracy_key=[f"eval/downstream/{key}_mc_5shot_acc" for key in v2_core_small_names]
        + [f"eval/downstream/{key}_mc_5shot_len_norm" for key in v2_mmlu_test_names],
        task_minimum=0.25,
        task_maximum=1.0,
        display_name="8 Task Average",
    ),
}


def get_task_sets(keys):
    if len(keys) == 1:
        if keys[0] == "core":
            keys = core_5shot_tasks.keys()
        elif keys[0] == "core_small_avg":
            keys = ["core_small_avg_5shot"]
        elif keys[0] == "mmlu":
            keys = list(mmlu_var_tasks.keys()) + list(mmlu_subset_var_tasks.keys())
        elif keys[0] == "mmlu_subset":
            keys = list(mmlu_subset_var_tasks.keys())
        elif keys[0] == "main":
            keys = list(mmlu_var_tasks.keys()) + list(core_small_5shot_tasks.keys())
        elif keys[0] == "main_mc":
            keys = ["mmlu_avg_var", "arc_challenge_5shot"]
        elif keys[0] == "all":
            keys = list(mmlu_var_tasks.keys()) + list(core_5shot_tasks.keys())
        elif keys[0] == "v2_main":
            keys = list(v2_mmlu_avg_test_5shot_tasks.keys()) + list(
                v2_core_small_5shot_tasks.keys()
            )
        elif keys[0] == "v2_main_variance":
            keys = list(v2_mmlu_avg_test_5shot_tasks.keys()) + list(v2_core_5shot_tasks.keys())
            keys = [
                k
                for k in keys
                if k
                not in ["openbookqa_val_5shot", "arc_challenge_val_5shot", "arc_easy_val_5shot"]
            ]
        elif keys[0] == "v2_main_avg":
            keys = ["main_avg_5shot"]
    return keys


def get_bpb_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    bpb_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_loss_key, list):
            bpb_keys += task.task_loss_key
        else:
            bpb_keys.append(task.task_loss_key)
    return bpb_keys


def get_accuracy_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    accuracy_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_accuracy_key, list):
            accuracy_keys += task.task_accuracy_key
        else:
            accuracy_keys.append(task.task_accuracy_key)
    return accuracy_keys


def get_mc_accuracy_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    mc_accuracy_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_mc_accuracy_key, list):
            mc_accuracy_keys += task.task_mc_accuracy_key
        else:
            mc_accuracy_keys.append(task.task_mc_accuracy_key)
    return mc_accuracy_keys


def get_soft_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    soft_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_soft_loss_key, list):
            soft_keys += task.task_soft_loss_key
        else:
            soft_keys.append(task.task_soft_loss_key)
    return soft_keys


def get_log_soft_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    log_soft_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_log_soft_loss_key, list):
            log_soft_keys += task.task_log_soft_loss_key
        else:
            log_soft_keys.append(task.task_log_soft_loss_key)
    return log_soft_keys


# Special case for testing with old tokenizer:

downstream_newline = [
    "mmlu_newline_social_sciences_var_len_norm",
    "mmlu_newline_humanities_var_len_norm",
    "mmlu_newline_other_var_len_norm",
    "mmlu_newline_stem_mc_5shot_test_len_norm",
    "mmlu_newline_humanities_mc_5shot_len_norm",
    "mmlu_newline_social_sciences_mc_5shot_len_norm",
    "mmlu_newline_stem_var_len_norm",
    "mmlu_newline_other_mc_5shot_test_len_norm",
    "mmlu_newline_humanities_mc_5shot_test_len_norm",
    "mmlu_newline_stem_mc_5shot_len_norm",
    "mmlu_newline_social_sciences_mc_5shot_test_len_norm",
    "mmlu_newline_other_mc_5shot_len_norm",
    "hellaswag_newline_rc_0shot_len_norm",
    "hellaswag_newline_rc_5shot_len_norm",
    "hellaswag_newline_mc_5shot_acc",
    "winogrande_newline_rc_0shot_acc",
    "winogrande_newline_rc_5shot_acc",
    "winogrande_newline_mc_5shot_acc",
    "piqa_newline_rc_0shot_len_norm",
    "piqa_newline_rc_5shot_len_norm",
    "piqa_newline_mc_5shot_acc",
    "socialiqa_newline_rc_0shot_len_norm",
    "socialiqa_newline_rc_5shot_len_norm",
    "socialiqa_newline_mc_5shot_acc",
    "openbookqa_newline_rc_0shot_len_norm",
    "openbookqa_newline_rc_5shot_len_norm",
    "openbookqa_newline_mc_5shot_acc",
    "csqa_newline_rc_0shot_len_norm",
    "csqa_newline_rc_5shot_len_norm",
    "csqa_newline_mc_5shot_acc",
    "boolq_newline_rc_0shot_acc",
    "boolq_newline_rc_5shot_acc",
    "boolq_newline_mc_5shot_acc",
    "copa_newline_rc_0shot_acc",
    "arc_easy_newline_rc_0shot_acc",
    "arc_easy_newline_rc_5shot_acc",
    "arc_easy_newline_mc_5shot_acc",
    "arc_challenge_newline_rc_0shot_len_norm",
    "arc_challenge_newline_rc_5shot_len_norm",
    "arc_challenge_newline_mc_5shot_acc",
    "sciq_newline_rc_0shot_acc",
]
downstream_newline_bpb = [
    "mmlu_newline_stem_var_bpb",
    "mmlu_newline_humanities_var_bpb",
    "mmlu_newline_social_sciences_var_bpb",
    "mmlu_newline_other_var_bpb",
    "mmlu_newline_stem_bpb",
    "mmlu_newline_humanities_bpb",
    "mmlu_newline_social_sciences_bpb",
    "mmlu_newline_other_bpb",
    "piqa_newline_rc_0shot_bpb",
    "piqa_newline_rc_5shot_bpb",
    "piqa_newline_mc_5shot_bpb",
    "hellaswag_newline_rc_0shot_bpb",
    "hellaswag_newline_rc_5shot_bpb",
    "hellaswag_newline_mc_5shot_bpb",
    "winogrande_newline_rc_0shot_bpb",
    "winogrande_newline_rc_5shot_bpb",
    "winogrande_newline_mc_5shot_bpb",
    "openbookqa_newline_rc_0shot_bpb",
    "openbookqa_newline_rc_5shot_bpb",
    "openbookqa_newline_mc_5shot_bpb",
    "boolq_newline_rc_0shot_bpb",
    "boolq_newline_rc_5shot_bpb",
    "boolq_newline_mc_5shot_bpb",
    "sciq_newline_rc_0shot_bpb",
    # "sciq_newline_rc_5shot_bpb",
    # "sciq_newline_mc_5shot_bpb",
    "arc_easy_newline_rc_0shot_bpb",
    "arc_easy_newline_rc_5shot_bpb",
    "arc_easy_newline_mc_5shot_bpb",
    "arc_challenge_newline_rc_0shot_bpb",
    "arc_challenge_newline_rc_5shot_bpb",
    "arc_challenge_newline_mc_5shot_bpb",
    "copa_newline_rc_0shot_bpb",
    # "copa_newline_rc_5shot_bpb",
    # "copa_newline_mc_5shot_bpb",
    "csqa_newline_rc_0shot_bpb",
    "csqa_newline_rc_5shot_bpb",
    "csqa_newline_mc_5shot_bpb",
    "socialiqa_newline_rc_0shot_bpb",
    "socialiqa_newline_rc_5shot_bpb",
    "socialiqa_newline_mc_5shot_bpb",
]

v1_tasks = {
    **mmlu_var_tasks,
    **mmlu_subset_var_tasks,
    **core_5shot_tasks,
    **core_small_avg_5shot_tasks,
}
v2_tasks = {**v2_mmlu_avg_val_5shot_tasks, **v2_mmlu_avg_test_5shot_tasks, **v2_core_5shot_tasks}
tasks = {**v1_tasks, **v2_tasks, **v2_main_avg_5shot_tasks}

downstream_bpb = get_bpb_keys({**mmlu_var_tasks, **core_5shot_tasks})
downstream = get_accuracy_keys({**mmlu_var_tasks, **core_5shot_tasks}) + get_mc_accuracy_keys(
    {**mmlu_var_tasks, **core_5shot_tasks}
)
v2_downstream_bpb = get_bpb_keys(v2_tasks)
v2_downstream_soft = get_soft_keys(v2_tasks)
v2_downstream_soft_log = get_log_soft_keys(v2_tasks)
v2_downstream_rc_acc = get_accuracy_keys(v2_tasks)
v2_downstream_mc_acc = get_mc_accuracy_keys(v2_tasks)

KEYS_BY_KEY = {
    "all-val-lm": [f"eval/{val}/CrossEntropyLoss" for val in validation],
    "all-bpb": downstream_bpb,
    "c4": ["eval/c4_en-validation/CrossEntropyLoss"],
}
for task_name, task in tasks.items():
    KEYS_BY_KEY[task_name] = (
        task.task_loss_key if isinstance(task.task_loss_key, list) else [task.task_loss_key]
    )

WEIGHT_BY_KEY = {
    "eval/downstream_bpb/mmlu_stem_var_bpb_bpb": 0.215,
    "eval/downstream_bpb/mmlu_humanities_var_bpb_bpb": 0.335,
    "eval/downstream_bpb/mmlu_social_sciences_var_bpb_bpb": 0.219,
    "eval/downstream_bpb/mmlu_other_var_bpb_bpb": 0.231,
    "eval/downstream/mmlu_stem_var_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_var_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_var_len_norm": 0.219,
    "eval/downstream/mmlu_other_var_len_norm": 0.231,
    "eval/downstream/mmlu_stem_mc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_mc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_mc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_mc_5shot_len_norm": 0.231,
    "eval/downstream_bpb/mmlu_stem_val_rc_5shot_bpb": 0.215,
    "eval/downstream_bpb/mmlu_humanities_val_rc_5shot_bpb": 0.335,
    "eval/downstream_bpb/mmlu_social_sciences_val_rc_5shot_bpb": 0.219,
    "eval/downstream_bpb/mmlu_other_val_rc_5shot_bpb": 0.231,
    "eval/downstream_bpb/mmlu_stem_test_rc_5shot_bpb": 0.215,
    "eval/downstream_bpb/mmlu_humanities_test_rc_5shot_bpb": 0.335,
    "eval/downstream_bpb/mmlu_social_sciences_test_rc_5shot_bpb": 0.219,
    "eval/downstream_bpb/mmlu_other_test_rc_5shot_bpb": 0.231,
    "eval/downstream_soft/mmlu_stem_val_rc_5shot_soft": 0.215,
    "eval/downstream_soft/mmlu_humanities_val_rc_5shot_soft": 0.335,
    "eval/downstream_soft/mmlu_social_sciences_val_rc_5shot_soft": 0.219,
    "eval/downstream_soft/mmlu_other_val_rc_5shot_soft": 0.231,
    "eval/downstream_soft/mmlu_stem_test_rc_5shot_soft": 0.215,
    "eval/downstream_soft/mmlu_humanities_test_rc_5shot_soft": 0.335,
    "eval/downstream_soft/mmlu_social_sciences_test_rc_5shot_soft": 0.219,
    "eval/downstream_soft/mmlu_other_test_rc_5shot_soft": 0.231,
    "eval/downstream_soft_log/mmlu_stem_val_rc_5shot_soft_log": 0.215,
    "eval/downstream_soft_log/mmlu_humanities_val_rc_5shot_soft_log": 0.335,
    "eval/downstream_soft_log/mmlu_social_sciences_val_rc_5shot_soft_log": 0.219,
    "eval/downstream_soft_log/mmlu_other_val_rc_5shot_soft_log": 0.231,
    "eval/downstream_soft_log/mmlu_stem_test_rc_5shot_soft_log": 0.215,
    "eval/downstream_soft_log/mmlu_humanities_test_rc_5shot_soft_log": 0.335,
    "eval/downstream_soft_log/mmlu_social_sciences_test_rc_5shot_soft_log": 0.219,
    "eval/downstream_soft_log/mmlu_other_test_rc_5shot_soft_log": 0.231,
    "eval/downstream/mmlu_stem_val_rc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_val_rc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_val_rc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_val_rc_5shot_len_norm": 0.231,
    "eval/downstream/mmlu_stem_test_rc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_test_rc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_test_rc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_test_rc_5shot_len_norm": 0.231,
    "eval/downstream/mmlu_stem_val_mc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_val_mc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_val_mc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_val_mc_5shot_len_norm": 0.231,
    "eval/downstream/mmlu_stem_test_mc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_test_mc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_test_mc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_test_mc_5shot_len_norm": 0.231,
}
