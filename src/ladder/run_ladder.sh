./src/ladder/ladder-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./src/ladder/ladder-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./src/ladder/ladder-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./src/ladder/ladder-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128


./src/ladder/ladder-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 4 --batch_size_divisor 128 --alpha_f 1.0
./src/ladder/ladder-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./src/ladder/ladder-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --alpha_f 1.0


./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 0.5xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 50 --eval_interval 50
./src/ladder/ladder-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 0.5xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 50 --eval_interval 50
./src/ladder/ladder-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 0.5xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 50 --eval_interval 50
./src/ladder/ladder-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 0.5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4 --save_interval 50 --eval_interval 50
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 0.5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 50 --eval_interval 50

./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval-rerun --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 100 --eval_interval 100
./src/ladder/ladder-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval-rerun --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 100 --eval_interval 100
./src/ladder/ladder-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval-rerun --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 100 --eval_interval 100
./src/ladder/ladder-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 1xC --name peteish-moreeval-rerun --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4 --save_interval 100 --eval_interval 100
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 1xC --name peteish-moreeval-rerun --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 100 --eval_interval 100

./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./src/ladder/ladder-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./src/ladder/ladder-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8
./src/ladder/ladder-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4

./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 500 --eval_interval 500
./src/ladder/ladder-launch.sh 4 --model 370M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 500 --eval_interval 500
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 500 --eval_interval 500
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 500 --eval_interval 500
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 500 --eval_interval 500

./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000
./src/ladder/ladder-launch.sh 4 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000
./src/ladder/ladder-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 1000 --eval_interval 1000
./src/ladder/ladder-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 1000 --eval_interval 1000
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 1000 --eval_interval 1000


./src/ladder/ladder-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval-const --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000 --alpha_f 1.0
./src/ladder/ladder-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval-const --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000 --alpha_f 1.0
./src/ladder/ladder-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval-const --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 1000 --eval_interval 1000 --alpha_f 1.0
./src/ladder/ladder-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish-moreeval-const --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4 --save_interval 1000 --eval_interval 1000 --alpha_f 1.0
./src/ladder/ladder-launch.sh 16 --model 3B --data olmoe-mix-0924 --length 10xC --name peteish-moreeval-const --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 1000 --eval_interval 1000 --alpha_f 1.0
