#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet/code
nohup python train.py \
--mode run_resume \
--test_dataset '/home/guozy/BISHE/dataset/BSD100' \
--nEpochs 210 \
--G_lr 1e-5 \
--D_lr 1e-5 \
--train_crop_size 128 \
--test_crop_size 320 \
--checkpoint '/home/guozy/BISHE/MyNet/result/on_MyDataset/checkpoints/5003_checkpoint.pkl' \
>> nohup.out 2>&1 &