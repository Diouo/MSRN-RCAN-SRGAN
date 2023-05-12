#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet_ddp/code
# nohup torchrun --nproc_per_node 2 \
nohup python -m torch.distributed.launch --nproc_per_node 2 \
train_ddp.py \
--mode pretrain_resume \
--train_dataset 'MyDataset' \
--test_dataset 'MyDataset' \
--G_pretrain_epoch 100 \
--G_lr 1e-5 \
--checkpoint '/home/guozy/BISHE/MyNet/result/weight/checkpoints/4050_checkpoint.pkl' \
--train_crop_size 128 \
--test_crop_size 320 \
>> nohup.out 2>&1 &