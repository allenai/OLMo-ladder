
# Ladder

The ladder is a set of small model configurations with the OLMo architecture, from sizes 190M to 3B, all trained to varying lengths expressed by Chinchilla-optimal multipliers. The code is based on the older [OLMo](https://github.com/allenai/olmo) codebase. The newer version using [OLMo-core](https://github.com/allenai/olmo-core) is under construction.


## Launch a training run

This will train the model, and run in-loop evaluations.

```bash
torchrun [OPTS..] src/ladder/ladder.py train
    --model 190M
    --length 1xC
    --name olmo-2-ladder-190M-1xC
    --device_batch_size 4
```

## Adding new evaluation sets

New evals can be added to [ladder.py](ladder.py), and then to [src/scaling/utils.py](../scaling/utils.py) for fitting scaling laws.

## Backfill new evaluations for existing ladder run

This will only run (new) evaluations for an already trained model.

```bash
torchrun [OPTS..] src/ladder/ladder.py eval 
    --model 190M
    --length 1xC
    --name olmo-2-ladder-190M-1xC-new-evals
    --device_batch_size 4
    --load_path saved_models/olmo-2-ladder-190M-1xC/step7272-unsharded
```

## Beaker usage

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