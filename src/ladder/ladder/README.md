
# Ladder

The ladder is a set of small model configurations with the OLMo architecture, from sizes 190M to 3B.

WIP: add instructions on how to update model config, etc.

TODO: remove dependency on old OLMo code.

## Beaker usage

### Launch a training run

TODO: run olmo-core ladder code, compare against old code, replace old code with olmo-core.

```bash
./src/ladder/ladder-launch.sh 4 \
    --model 190M \
    --data olmoe-mix-0924 \
    --length 1xC \
    --name peteish \
    --save_overwrite \
    --device_batch_size 4 \
    --batch_size_divisor 128
```

### Backfill new evaluations for existing ladder run

```bash
./src/ladder/ladder_eval-launch.sh 2 \
    --model 190M \
    --data olmoe-mix-0924 \
    --length 1xC \
    --name peteish-final-eval \
    --save_overwrite \
    --device_batch_size 4 \
    --batch_size_divisor 64 \
    --device_eval_batch_size 16 \
    --load_path /weka/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-final-190M-1xC/step7272-unsharded
```

## Adding new evaluation sets

New evals can be added to [ladder.py](ladder.py), and then to [src/scaling/utils.py](../scaling/utils.py) for fitting scaling laws.

TODO: add instructions when we shift to using OLMo-core.


### Variance analysis

For new evaluation sets, variance analysis can be done as follows:

1. Backfill results for the 1B-10xC ladder model (as shown above).
2. Run analysis as follows:

```bash
python src/scripts/variance_analysis.py \
    -k <new_eval_set> \
    -c src/scripts/paper/configs/final_variance.json \
    -o figures/variance.png \
    --last_n_points 10 \
    --run_prediction \
    --print_table_as_latex
```
