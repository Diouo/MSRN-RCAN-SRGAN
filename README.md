# MSRN-RCAN-SRGAN
This repository contains an simple PyTorch implementation of [SRGAN](https://arxiv.org/abs/1609.04802) which is added with
    [EDSR](https://arxiv.org/abs/1707.02921),
    [RCAN](https://arxiv.org/abs/1807.02758), 
    [MSRN](https://arxiv.org/abs/1904.10698)

Attention!:\
    The function resize of torch is not bicubic by default\
    psnr and ssim calculations are generally on the y channel\
    Use float32 for psnr calculation and uint8 for ssim calculation\
    