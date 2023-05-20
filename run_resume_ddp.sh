#!/usr/bin/bash
source activate
conda activate cyan_torch
cd /home/guozy/BISHE/MyNet_ddp/code
# nohup torchrun --nproc_per_node 2 \
nohup python -m torch.distributed.launch --nproc_per_node 2 \
train_ddp.py \
--mode run_resume \
--test_dataset 'BSD100' \
--nEpochs 10000 \
--G_lr 1e-5 \
--D_lr 1e-5 \
--train_crop_size 128 \
--test_crop_size 320 \
--batchSize 16 \
--checkpoint '/home/guozy/BISHE/MyNet_ddp/result/GAN/checkpoints/10050_checkpoint.pkl' \
--model_out_path '/home/guozy/BISHE/MyNet_ddp/result/GAN' \
>> nohup.out 2>&1 &