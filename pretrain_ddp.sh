#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet/code
# nohup torchrun --nproc_per_node 2 \
nohup python -m torch.distributed.launch --nproc_per_node 2 \
train_ddp.py \
--mode pretrain \
--test_dataset 'BSD100' \
--G_pretrain_epoch 2000 \
--G_lr 1e-4 \
--train_crop_size 128 \
--test_crop_size 320 \
>> nohup.out 2>&1 &