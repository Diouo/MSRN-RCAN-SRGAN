import os
import time
import datetime
import numpy as np
import random
import argparse

import torch
import torch.backends.cudnn as cudnn

from net.solver_ddp import MyNetTrainer


# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')

# Ttraining mode settings
parser.add_argument('--mode', type=str, default='run', help='pretrain/pretrain_resume/run/run_resume')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--model_out_path', type=str, default=None)
parser.add_argument("--local_rank", default=-1, type=int)

# dataset settings
parser.add_argument('--train_dataset', type=str, default='DIV2K', help='desicion of dataset')
parser.add_argument('--test_dataset', type=str, default='DIV2K', help='desicion of dataset')
parser.add_argument('--train_crop_size', type=int, default=128, help='crop size of the sample')
parser.add_argument('--test_crop_size', type=int, default=256, help='crop size of the sample')
parser.add_argument('--test_image', type=int, default='/home/guozy/BISHE/dataset/Set14/comic.png', help='for show resolve')

# hyper-parameters
parser.add_argument('--num_residuals', type=int, default=23)
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--G_pretrain_epoch', type=int, default=400)
parser.add_argument('--G_lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--D_lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--D_threshold', type=float, default=0.4, help='use for limit the training of D')
parser.add_argument('--seed', type=int, default=42, help='random seed to use')
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="only 2 * k")

args = parser.parse_args()


def main():
    # ===========================================================
    # prepare for store model
    # ===========================================================
    if args.model_out_path is None:
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H:%M:%S")
        model_out_path = '/home/guozy/BISHE/MyNet_ddp/result/' + now
        if os.path.exists(model_out_path) == False:
                os.mkdir(model_out_path)

        checkpoints_out_path = model_out_path +'/checkpoints/'
        if os.path.exists(checkpoints_out_path) == False:
            os.mkdir(checkpoints_out_path)

    else:
        model_out_path = args.model_out_path
        checkpoints_out_path = model_out_path +'/checkpoints/'

    # ===========================================================
    # To store settings information of model
    # ===========================================================
    argsDict = args.__dict__
    with open(model_out_path + '/information.txt', 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    # ===========================================================
    # train model
    # ===========================================================
    time_start = time.time()
    model = MyNetTrainer(args, model_out_path)
    if args.mode == 'run':
        best_psnr, best_ssim, best_epoch = model.run()
    elif args.mode == 'run_resume':
        best_psnr, best_ssim, best_epoch = model.run_resume()
    elif args.mode == 'pretrain':
        best_psnr, best_ssim, best_epoch = model.pretrain()
    elif args.mode == 'pretrain_resume':
        best_psnr, best_ssim, best_epoch = model.pretrain_resume()
         
    # ===========================================================
    # check how much time was used to train model
    # ===========================================================
    time_end = time.time()
    print('\n===> best_psnr:{}, best_ssim:{}, best_epoch:{}'.format(best_psnr, best_ssim, best_epoch))
    print('\n===> time cost:{}s'.format( time_end - time_start))  
    with open(model_out_path + '/information.txt', 'a') as f:
        f.write('\nbest_psnr:{}, best_ssim:{}, best_epoch:{}'.format(best_psnr, best_ssim, best_epoch))
        f.write('\ntime cost:{}s\n\n'.format(time_end - time_start))
        f.close()


if __name__ == '__main__':

    # set the random number seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    # set for DDP
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    main()
