#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet/code
nohup python train.py \
--mode pretrain_resume \
--batchSize 16 \
--G_pretrain_epoch 5500 \
--G_lr 1e-5 \
--checkpoint '/home/guozy/BISHE/MyNet/result/CNN/checkpoints/7200_checkpoint.pkl' \
--model_out_path '/home/guozy/BISHE/MyNet/result/CNN' \
--train_crop_size 128 \
--test_crop_size 320 \
>> nohup.out 2>&1 &