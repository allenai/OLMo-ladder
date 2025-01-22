
# Ladder

The ladder is a set of small model configurations.

TODO: add details

TODO: use olmo-core instead of old olmo scripts.

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