#!/usr/bin/bash
source activate
conda activate bishe
cd /home/guozy/BISHE/MyNet/code
nohup python train.py \
--mode run \
--test_dataset '/home/guozy/BISHE/dataset/BSD100' \
--nEpochs 500 \
--G_lr 1e-4 \
--D_lr 1e-4 \
--checkpoint '/home/guozy/BISHE/MyNet/result/weight/734_weight.pkl' \
>> nohup.out 2>&1 &