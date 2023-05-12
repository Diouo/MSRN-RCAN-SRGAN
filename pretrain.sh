#!/usr/bin/bash
source activate
conda activate bishe
cd /home/guozy/BISHE/MyNet/code
nohup python train.py \
--mode pretrain \
--G_pretrain_epoch 2000 \
--G_lr 1e-4 \
--test_dataset 'BSD100' \
--train_crop_size 128 \
--test_crop_size 256 \
>> nohup.out 2>&1 &