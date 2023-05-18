# MSRN-RCAN-SRGAN
This repository contains an simple PyTorch implementation of [SRGAN](https://arxiv.org/abs/1609.04802) which is combined with
    [EDSR](https://arxiv.org/abs/1707.02921),
    [RCAN](https://arxiv.org/abs/1807.02758), 
    [MSRN](https://arxiv.org/abs/1904.10698)


## Code Attention!:\
The function resize of torch is not bicubic by default,but the default has been change to Bicubic now\
You have to assgin your own lr_scheduler in net.solver.__init__\
D_train require D_loss less than 0.4, you can change in net.solver.D_train !!!\
Hyper parameter K in GAN is not supported(default==1) because it is useless in practice, you have to recode net.solver.run/run_resume if you want \


## Test Attention!:\
PSNR and SSIM calculations are only on the y channel\
Use float32 for PSNR calculation and uint8 for SSIM calculation\
