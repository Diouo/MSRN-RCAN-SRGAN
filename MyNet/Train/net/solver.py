from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg19

from net.model import Generator, Discriminator
from .progress_bar import progress_bar


class MyNetTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(MyNetTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.netG = None
        self.netD = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.epoch_pretrain = 10
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None
        self.feature_extractor = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.num_residuals = 16
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64, num_channel=3).to(self.device)
        self.netD = Discriminator(base_filter=64, num_channel=3).to(self.device)
        self.feature_extractor = vgg19(weights='IMAGENET1K_V1')
        self.netG.weight_init(mean=0.0, std=0.2)
        self.netD.weight_init(mean=0.0, std=0.2)
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        
        if self.GPU_IN_USE:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.feature_extractor.cuda()
            self.criterionG.cuda()
            self.criterionD.cuda()

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr / 100)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 75, 100], gamma=0.5)  # lr decay
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 75, 100], gamma=0.5)  # lr decay


    def save(self, model_out_path):
        g_model_out_path = model_out_path + "/MyNet_Generator_model_path.pth"
        d_model_out_path = model_out_path + "/MyNet_Discriminator_model_path.pth"
        torch.save(self.netG, g_model_out_path)
        torch.save(self.netD, d_model_out_path)
        print("Checkpoint saved to {}".format(g_model_out_path))
        print("Checkpoint saved to {}".format(d_model_out_path))


    def train(self):
        # models setup
        self.netG.train()
        self.netD.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # setup noise
            real_label = torch.ones(data.size(0), 1).to(self.device)
            fake_label = torch.zeros(data.size(0), 1).to(self.device)
            data, target = data.to(self.device), target.to(self.device) # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])


            # Train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.netD(target) # 真实样本的判别概率
            d_real_loss = self.criterionD(d_real, real_label) # 真实样本的损失

            d_fake = self.netD(self.netG(data)) # 虚假样本的判别概率
            d_fake_loss = self.criterionD(d_fake, fake_label) # 虚假样本的损失

            d_total = d_real_loss + d_fake_loss # 总损失
            d_train_loss += d_total.item() # 为了可视化批与整个数据集的变量
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.netG(data) # 虚假样本, torch.Size([4, 3, 1024, 1024])
            g_fake = self.netD(g_real) # 虚假样本的判别概率
            gan_loss = self.criterionD(g_fake, real_label) # 虚假样本的生成损失
            mse_loss = self.criterionG(g_real, target) # 虚假样本的感知损失

            g_total = mse_loss + 1e-3 * gan_loss # 总损失
            g_train_loss += g_total.item() # 为了可视化批与整个数据集的变量
            g_total.backward()
            self.optimizerG.step()

            progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))

        print("    Average G_Loss: {:.4f}".format(g_train_loss / len(self.training_loader)))


    def test(self):
        self.netG.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.netG(data)

                mse = self.criterionG(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step()

        self.save()
