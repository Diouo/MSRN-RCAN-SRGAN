# MSRN-RCAN-SRGAN
This repository contains an simple PyTorch implementation of [SRGAN](https://arxiv.org/abs/1609.04802) combined with
    [EDSR](https://arxiv.org/abs/1707.02921),
    [RCAN](https://arxiv.org/abs/1807.02758), 
    [MSRN](https://arxiv.org/abs/1904.10698)\

Now CPU/OneGPU version has been implemented, you can use it freely with bash in folder: MyNet \
  
As for DDP version, the code is also available in folder: MyNet_ddp but may not be easy to use, and it will coming soon \
  
Pretrained weight for CNN and GAN also offer [Google Drive](https://drive.google.com/drive/folders/1SM0gsugcjTfYDlYrIqyWXFZyYuA6LkXc?usp=sharing)


## Code Attention:
1. The function resize of torch is not bicubic by default, but the default has been change to Bicubic now
2. You can assgin your own lr_scheduler in net.solver.__init__
3. D_train require d_real_loss/d_fake_loss less than 0.4, you can change the value in train.py !!!
3. Hyper parameter K in GAN is not supported(default==1) because it is useless in practice, you have to recode net.solver.run/run_resume if you want


## Test Attention:
1. PSNR and SSIM calculations are only on the y channel
2. Use float32 for PSNR calculation and uint8 for SSIM calculation
