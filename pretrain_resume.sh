#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet/code
nohup python train.py \
--mode pretrain_resume \
--train_dataset 'MyDataset' \
--test_dataset 'MyDataset' \
--batchSize 3 \
--G_pretrain_epoch 5500 \
--G_lr 1e-5 \
--checkpoint '/home/guozy/BISHE/MyNet/result/on_MyDataset/checkpoints/8250_checkpoint.pkl' \
--train_crop_size 128 \
--test_crop_size 320 \
>> nohup.out 2>&1 &