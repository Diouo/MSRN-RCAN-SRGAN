from math import log10
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg16
from torch.autograd import Variable

from MyNet.model import Generator, Discriminator
from .progress_bar import progress_bar


def calculate_gradient_penalty(self, real_images, fake_images):
    eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    if self.cuda:
        eta = eta.cuda(self.cuda_index)
    else:
        eta = eta

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    if self.cuda:
        interpolated = interpolated.cuda(self.cuda_index)
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
    return grad_penalty



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
        self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64, num_channel=1).to(self.device)
        self.netD = Discriminator(base_filter=64, num_channel=1).to(self.device)
        self.feature_extractor = vgg16(pretrained=True)
        self.netG.weight_init(mean=0.0, std=0.2)
        self.netD.weight_init(mean=0.0, std=0.2)
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            self.feature_extractor.cuda()
            cudnn.benchmark = True
            self.criterionG.cuda()
            self.criterionD.cuda()

        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.lr)
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.lr / 100)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 75, 100], gamma=0.5)  # lr decay
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    @staticmethod
    def to_data(x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def save(self, model_out_path):
        g_model_out_path = model_out_path + "/MyNet_Generator_model_path.pth"
        d_model_out_path = model_out_path + "/MyNet_Discriminator_model_path.pth"
        torch.save(self.netG, g_model_out_path)
        torch.save(self.netD, d_model_out_path)
        print("Checkpoint saved to {}".format(g_model_out_path))
        print("Checkpoint saved to {}".format(d_model_out_path))

    def pretrain(self):
        self.netG.train()
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.netG.zero_grad()
            loss = self.criterionG(self.netG(data), target)
            loss.backward()
            self.optimizerG.step()

    def train(self):
        # models setup
        self.netG.train()
        self.netD.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # setup noise
            real_label = torch.ones(data.size(0), data.size(1)).to(self.device)
            fake_label = torch.zeros(data.size(0), data.size(1)).to(self.device)
            data, target = data.to(self.device), target.to(self.device)

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
            g_real = self.netG(data) # 虚假样本
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

        # for epoch in range(1, self.epoch_pretrain + 1):
        #     self.pretrain()
        #     print("{}/{} pretrained".format(epoch, self.epoch_pretrain))

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()
