import random
import numpy as np
import os
from math import log10
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch import autograd
from torch.utils.tensorboard import SummaryWriter

from net.model import Generator, Discriminator, VGG19


class MyNetTrainer(object):
    def __init__(self, config, training_loader, testing_loader, model_out_path):
        super(MyNetTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.netG = None
        self.netD = None
        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        self.nEpochs = config.nEpochs
        self.criterionG = None
        self.criterionD = None
        self.criterionF= None
        self.optimizerG = None
        self.optimizerD = None
        self.feature_extractor = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.num_residuals = 16
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.model_out_path = model_out_path
        self.writer = SummaryWriter(model_out_path + '/tensorboard')


    def build_model(self):
        if self.GPU_IN_USE:
            # set the random number seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            
            # build Generator
            self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64, num_channel=3).to('cuda:0')
            self.netG.weight_init(mean=0.0, std=0.2)
            self.criterionG = nn.MSELoss()
            self.criterionG.to('cuda:0')
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.G_lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)  # lr decay
            self.writer.add_graph(self.netG, input_to_model=torch.randn(16, 3, 32, 32).to('cuda:0'), verbose=False)
            
            # build Discriminator
            self.netD = Discriminator(base_filter=64, num_channel=3).to('cuda:1')
            self.netD.weight_init(mean=0.0, std=0.2)
            self.criterionD = nn.BCELoss()
            self.criterionD.to('cuda:1')
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.D_lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)  # lr decay

            # build feature extractor
            self.feature_extractor = VGG19().to('cuda:0')


    def pretrain(self):
        print('\n===> Pretrain')
        for _, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            self.netG.eval()
            self.netD.train()
            self.optimizerD.zero_grad()

            real_label = torch.ones(data.size(0), 1)
            fake_label = torch.zeros(data.size(0), 1)

            d_real = self.netD(target.to('cuda:1')) # 真实样本的判别概率
            d_real_loss = self.criterionD(d_real, real_label.to('cuda:1')) # 真实样本的损失

            d_fake = self.netD(self.netG(data.to('cuda:0')).to('cuda:1')) # 虚假样本的判别概率
            d_fake_loss = self.criterionD(d_fake, fake_label.to('cuda:1')) # 虚假样本的损失

            d_total =  d_real_loss + d_fake_loss  # 总损失
            d_total.backward()
            self.optimizerD.step()


    def train(self,epoch):
        print('     Training')

        self.feature_extractor.eval()
        g_train_loss = 0
        d_train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            # setup noise
            real_label = torch.ones(data.size(0), 1)
            fake_label = torch.zeros(data.size(0), 1)

            # ===========================================================
            # Train Discriminator
            # ===========================================================
            self.netG.eval()
            self.netD.train()
            self.optimizerD.zero_grad()

            d_real = self.netD(target.to('cuda:1')) # 真实样本的判别概率
            d_real_loss = self.criterionD(d_real, real_label.to('cuda:1')) # 真实样本的损失

            d_fake = self.netD(self.netG(data.to('cuda:0')).to('cuda:1')) # 虚假样本的判别概率
            d_fake_loss = self.criterionD(d_fake, fake_label.to('cuda:1')) # 虚假样本的损失

            d_total =  d_real_loss + d_fake_loss  # 总损失
            d_total.backward()
            d_train_loss += d_total.item() # 为了可视化批与整个数据集的变量
            self.optimizerD.step()

            # ===========================================================
            # Train Generator
            # ===========================================================
            self.netG.train()
            self.netD.eval()
            self.optimizerG.zero_grad()

            g_real = self.netG(data.to('cuda:0')) # 虚假样本, torch.Size([4, 3, 1024, 1024])
            g_fake = self.netD(g_real.to('cuda:1')) # 虚假样本的判别概率
            gan_loss = self.criterionD(g_fake.to('cuda:0'), real_label.to('cuda:0')) # 虚假样本的对抗损失
            mse_loss = self.criterionG(g_real, target.to('cuda:0')) # 虚假样本的距离损失
            content_loss = self.feature_extractor.forward(g_real,target.to('cuda:0')) # 虚假样本的感知损失

            g_total = mse_loss + 1e-3 * gan_loss + 0.006 * content_loss# 总损失
            g_train_loss += g_total.item() # 为了可视化批与整个数据集的变量
            g_total.backward()
            self.optimizerG.step()
            
        self.writer.add_scalar(tag="train/G_loss", scalar_value=g_train_loss / (batch_num + 1), global_step=epoch)
        self.writer.add_scalar(tag="train/D_loss", scalar_value=d_train_loss / (batch_num + 1), global_step=epoch)


    def test(self,epoch):
        print('     Testing')

        self.netG.eval()
        avg_psnr = 0
        avg_ssim = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.testing_loader):
                data, target = data.to('cuda:0'), target.to('cuda:0')
                prediction = self.netG(data)

                mse = self.criterionG(prediction, target)
                avg_psnr += 10 * log10(1 / mse.item())
                avg_ssim += ssim(prediction.squeeze(dim=0).cpu().numpy(), target.squeeze(dim=0).cpu().numpy(), channel_axis=0) 
        
        img = Image.open('/home/guozy/BISHE/dataset/Set5/butterfly.png')
        data = (ToTensor()(img)) 
        data = data.to('cuda:0').unsqueeze(0) # torch.Size([1, 3, 256, 256])
        out = self.netG(data)
        out = out.detach().squeeze(0).clip(0, 255)

        self.writer.add_scalar(tag="test/PSNR", scalar_value=avg_psnr / len(self.testing_loader), global_step=epoch)
        self.writer.add_scalar(tag="test/SSIM", scalar_value=avg_ssim / len(self.testing_loader), global_step=epoch)
        self.writer.add_image("test/IMAGE", out, epoch, dataformats='CHW')

        return avg_psnr, avg_ssim


    def save(self, epoch):
        print('     Saving')
        checkpoint={
            'epoch':epoch,
            'D_state_dict':self.netD.state_dict(),
            'G_state_dict':self.netG.state_dict(),
            'optimize_state_dict_G':self.optimizerG.state_dict(),
            'optimize_state_dict_D':self.optimizerD.state_dict(),
                    }
        checkpoints_out_path = self.model_out_path +'/checkpoints/'
        if os.path.exists(checkpoints_out_path) == False:
            os.mkdir(checkpoints_out_path)
        torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')


    def run(self):
        best_psnr = 0
        best_ssim = 0
        best_epoch = 0

        self.build_model()
        self.pretrain()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts".format(epoch))

            self.train(epoch)
            temp_psnr, temp_ssim = self.test(epoch)
            self.scheduler.step()

            if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                best_psnr = temp_psnr
                best_ssim = temp_ssim
                best_epoch = epoch
                self.save(epoch)

        return best_epoch
    
    def resume(self):
        print('     Resuming')
