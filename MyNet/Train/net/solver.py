from math import log10
import random
import numpy as np
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
        self.lr = config.lr
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
            # print(next(self.netG.parameters()).device)
            self.netG.weight_init(mean=0.0, std=0.2)
            self.criterionG = nn.MSELoss()
            self.criterionG.to('cuda:0')
            self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[100, 200, 300, 400], gamma=0.5)  # lr decay
            
            # build Discriminator
            self.netD = Discriminator(base_filter=64, num_channel=3).to('cuda:1')
            self.netD.weight_init(mean=0.0, std=0.2)
            # print(next(self.netD.parameters()).device)
            self.criterionD = nn.BCELoss()
            self.criterionD.to('cuda:1')
            self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[100, 200, 300, 400], gamma=0.5)  # lr decay

            # build feature extractor
            self.feature_extractor = VGG19().to('cuda:0')


    def pretrain(self):
        print('\n===> Pretrain')
        for _, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            self.netG.eval()
            self.netD.train()
            self.optimizerD.zero_grad()

            d_real = self.netD(target.to('cuda:1')) # 真实样本的判别概率
            d_real_loss = torch.mean(d_real)

            d_fake = self.netD(self.netG(data.to('cuda:0')).to('cuda:1')) # 虚假样本的判别概率
            d_fake_loss = torch.mean(d_fake)

            # Train with gradient penalty
            g_real = self.netG(data.to('cuda:0')).to('cuda:1') # 虚假样本, torch.Size([4, 3, 1024, 1024])
            gradient_penalty = self.calculate_gradient_penalty(target.to('cuda:1').data, g_real.data)

            d_total =  -d_real_loss + d_fake_loss + gradient_penalty # 总损失
            d_total.backward()
            self.optimizerD.step()


    def train(self,epoch):
        print('     Training')

        self.feature_extractor.eval()
        g_train_loss = 0
        d_train_loss = 0
        Wasserstein_D = 0
        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            # setup noise
            # real_label = torch.ones(data.size(0), 1)
            # fake_label = torch.zeros(data.size(0), 1)

            # ===========================================================
            # Train Discriminator
            # ===========================================================
            self.netG.eval()
            self.netD.train()
            self.optimizerD.zero_grad()

            d_real = self.netD(target.to('cuda:1')) # 真实样本的判别概率
            # d_real_loss = self.criterionD(d_real, real_label.to('cuda:1')) # 真实样本的损失
            d_real_loss = torch.mean(d_real)

            d_fake = self.netD(self.netG(data.to('cuda:0')).to('cuda:1')) # 虚假样本的判别概率
            # d_fake_loss = self.criterionD(d_fake, fake_label.to('cuda:1')) # 虚假样本的损失
            d_fake_loss = torch.mean(d_fake)

            # Train with gradient penalty
            g_real = self.netG(data.to('cuda:0')).to('cuda:1') # 虚假样本, torch.Size([4, 3, 1024, 1024])
            gradient_penalty = self.calculate_gradient_penalty(target.to('cuda:1').data, g_real.data)

            d_total =  -d_real_loss + d_fake_loss + gradient_penalty * 10 # 总损失
            d_total.backward()
            d_train_loss += d_total.item() # 为了可视化批与整个数据集的变量
            Wasserstein_D += d_real_loss - d_fake_loss
            self.optimizerD.step()

            # ===========================================================
            # Train Generator
            # ===========================================================
            self.netG.train()
            self.netD.eval()
            self.optimizerG.zero_grad()

            g_real = self.netG(data.to('cuda:0')) # 虚假样本, torch.Size([4, 3, 1024, 1024])
            g_fake = self.netD(g_real.to('cuda:1')) # 虚假样本的判别概率
            # gan_loss = self.criterionD(g_fake.to('cuda:0'), real_label.to('cuda:0')) # 虚假样本的对抗损失
            gan_loss = torch.mean(g_fake.to('cuda:0'))
            mse_loss = self.criterionG(g_real, target.to('cuda:0')) # 虚假样本的距离损失
            content_loss = self.feature_extractor.forward(g_real,target.to('cuda:0')) # 虚假样本的感知损失

            g_total = mse_loss - 1e-3 * gan_loss + 0.006 * content_loss# 总损失
            g_train_loss += g_total.item() # 为了可视化批与整个数据集的变量
            g_total.backward()
            self.optimizerG.step()
            
        self.writer.add_scalar(tag="train/G_loss", scalar_value=g_train_loss / (batch_num + 1), global_step=epoch)
        self.writer.add_scalar(tag="train/D_loss", scalar_value=d_train_loss / (batch_num + 1), global_step=epoch)
        self.writer.add_scalar(tag="train/Wasserstein_distance", scalar_value=Wasserstein_D / (batch_num + 1), global_step=epoch)


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
        g_model_out_path = self.model_out_path +'/checkpoints/epoch_' + str(epoch) + "/Generator_model_path.pkl"
        d_model_out_path = self.model_out_path +'/checkpoints/epoch_' + str(epoch) +"/Discriminator_model_path.pkl"

        checkpoint_G={'modle':Generator(),
             'model_state_dict':self.netG.state_dict(),
             'optimize_state_dict':self.optimizerG.state_dict(),
             'epoch':epoch}
        
        checkpoint_D={'modle':Discriminator(),
             'model_state_dict':self.netD.state_dict(),
             'optimize_state_dict':self.optimizerD.state_dict(),
             'epoch':epoch}

        torch.save(checkpoint_G, g_model_out_path)
        torch.save(checkpoint_D, d_model_out_path)


    def run(self):
        best_psnr = 0
        best_ssim = 0

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
                self.save(epoch)


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(16,1,1,1).uniform_(0,1)
        eta = eta.expand(16, real_images.size(1), real_images.size(2), real_images.size(3)).to('cuda:1')
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
                                    outputs=prob_interpolated,
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to('cuda:1'),
                                    create_graph=True, retain_graph=True
                                )[0]

        # print(gradients.norm(2, dim=1).size())
        # grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradients = gradients.reshape(16,-1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return grad_penalty
