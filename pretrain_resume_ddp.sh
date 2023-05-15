#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet_ddp/code
# nohup torchrun --nproc_per_node 2 \
nohup python -m torch.distributed.launch --nproc_per_node 2 \
train_ddp.py \
--mode pretrain_resume \
--batchSize 8 \
--G_pretrain_epoch 6100 \
--G_lr 1e-5 \
--checkpoint '/home/guozy/BISHE/MyNet/result/CNN/checkpoints/4000_checkpoint.pkl' \
--model_out_path '/home/guozy/BISHE/MyNet_ddp/result/CNN' \
--train_crop_size 128 \
--test_crop_size 320 \
>> nohup.out 2>&1 &