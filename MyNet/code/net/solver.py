import os
from math import log10
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

from net.model import Generator, Discriminator, VGG19


class MyNetTrainer(object):
    def __init__(self, config, training_loader, testing_loader, model_out_path):
        super(MyNetTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.upscale_factor = config.upscale_factor
        self.nEpochs = config.nEpochs
        self.G_pretrain_epoch= config.G_pretrain_epoch
        self.num_residuals = 16

        self.netG = None
        self.netD = None
        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        self.criterionG = None
        self.criterionD = None
        self.criterionF= None
        self.optimizerG = None
        self.optimizerD = None
        self.schedulerG = None
        self.schedulerD = None
        self.feature_extractor = None
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        
        self.model_out_path = model_out_path
        self.checkpoint = config.checkpoint
        self.weight = config.weight
        self.writer = None
        

    def build_model(self):
        print('\n===> Building the Model')

        # build Generator
        self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64, num_channel=3).to('cuda:0')
        self.netG.weight_init(mean=0.0, std=0.2)
        self.criterionG = nn.MSELoss().to('cuda:0')
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.G_lr)
        
        # build Discriminator
        self.netD = Discriminator(base_filter=64, num_channel=3).to('cuda:0')
        self.netD.weight_init(mean=0.0, std=0.2)
        self.criterionD = nn.BCELoss().to('cuda:0')
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.D_lr)

        # build feature extractor
        self.feature_extractor = VGG19().to('cuda:0')


    def G_pretrain(self):
        print('     Training')
        for _, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])

            data = data.to('cuda:0')
            target = target.to('cuda:0')

            self.netG.train()
            self.netD.eval()
            self.optimizerG.zero_grad()

            g_real = self.netG(data) # fake samples, torch.Size([4, 3, 1024, 1024])
            mse_loss = self.criterionG(g_real, target) # MSE loss of fake samples
            content_loss = self.feature_extractor.forward(g_real,target) # VGG loss of fake samples

            g_total = mse_loss + 0.006 * content_loss # total loss of G
            g_total.backward()
            self.optimizerG.step()


    def train(self,epoch):
        print('     Training')

        self.feature_extractor.eval()
        g_train_loss = 0
        d_train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])

            # setup noise
            data = data.to('cuda:0')
            target = target.to('cuda:0')
            real_label = torch.ones(data.size(0), 1).to('cuda:0')
            fake_label = torch.zeros(data.size(0), 1).to('cuda:0')

            # ===========================================================
            # Train Discriminator
            # ===========================================================
            self.netG.eval()
            self.netD.train()
            self.optimizerD.zero_grad()

            d_real = self.netD(target) # prob of real samples
            d_real_loss = self.criterionD(d_real, real_label) # BCE loss of real samples

            d_fake = self.netD(self.netG(data)) # prob of fake samples
            d_fake_loss = self.criterionD(d_fake, fake_label) # BCE loss of fake samples

            d_total =  d_real_loss + d_fake_loss  # total loss of D
            d_total.backward()
            d_train_loss += d_total.item()
            self.optimizerD.step()

            # ===========================================================
            # Train Generator
            # ===========================================================
            self.netG.train()
            self.netD.eval()
            self.optimizerG.zero_grad()

            g_real = self.netG(data) # fake samples, torch.Size([4, 3, 1024, 1024])
            g_fake = self.netD(g_real) # prob of fake samples
            gan_loss = self.criterionD(g_fake, real_label) # Adversarial loss of fake samples
            mse_loss = self.criterionG(g_real, target) # MSE loss of fake samples
            content_loss = self.feature_extractor.forward(g_real,target) # VGG loss of fake samples

            g_total = mse_loss + 1e-3 * gan_loss + 0.006 * content_loss # total loss of G
            g_train_loss += g_total.item()
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
                data = data.to('cuda:0'),
                target = target.to('cuda:0')
                prediction = self.netG(data[0])
                mse = self.criterionG(prediction, target)
                avg_psnr += 10 * log10(1 / mse.item())
                avg_ssim += ssim(prediction.squeeze(dim=0).cpu().numpy(), target.squeeze(dim=0).cpu().numpy(), channel_axis=0) 
        
        img = Image.open('/home/guozy/BISHE/dataset/Set5/butterfly.png')
        data = (ToTensor()(img)) 
        data = data.to('cuda:0').unsqueeze(0) # torch.Size([1, 3, 256, 256])
        out = self.netG(data).detach().squeeze(0) # torch.Size([3, 1024, 1024])

        self.writer.add_scalar(tag="test/PSNR", scalar_value=avg_psnr / len(self.testing_loader), global_step=epoch)
        self.writer.add_scalar(tag="test/SSIM", scalar_value=avg_ssim / len(self.testing_loader), global_step=epoch)
        self.writer.add_image("test/IMAGE", out, epoch, dataformats='CHW')

        return avg_psnr, avg_ssim


    def save(self, best_psnr, best_ssim, best_epoch):
        print('     Saving')
        checkpoint={
            'epoch':best_epoch,
            'D_state_dict':self.netD.state_dict(),
            'G_state_dict':self.netG.state_dict(),
            'optimizeG_state_dict':self.optimizerG.state_dict(),
            'optimizeD_state_dict':self.optimizerD.state_dict(),
            # new add, not in baseline
            'schedulerG_state_dict':self.schedulerG.state_dict(),
            'schedulerD_state_dict':self.schedulerD.state_dict(),
            'best_psnr':best_psnr,
            'best_ssim':best_ssim,
                    }
        checkpoints_out_path = self.model_out_path +'/checkpoints/'
        if os.path.exists(checkpoints_out_path) == False:
            os.mkdir(checkpoints_out_path)
        torch.save(checkpoint, checkpoints_out_path + str(best_epoch) + '_checkpoint.pkl')

    
    def pretrain(self):
        best_psnr = 0
        best_ssim = 0
        best_epoch = 0
        self.build_model()
        self.writer = SummaryWriter(self.model_out_path + '/tensorboard')
        self.writer.add_graph(self.netG, input_to_model=torch.randn(1, 3, 32, 32).to('cuda:0'), verbose=False)

        self.schedulerG = None

        for epoch in range(1, self.G_pretrain_epoch + 1):
            print('\n===> G Pretraining Epoch {} starts'.format(epoch))
            self.G_pretrain()
            temp_psnr, temp_ssim = self.test(epoch)
            if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                best_psnr = temp_psnr
                best_ssim = temp_ssim
                best_epoch = epoch

                print('     Saving')
                weight = {'weight':self.netG.state_dict()}
                torch.save(weight, self.model_out_path + '/' + str(best_epoch) + '_weight.pkl')
                
        print('     Saving')
        weight = {'weight':self.netG.state_dict()}
        torch.save(weight, self.model_out_path + '/' + str(self.G_pretrain_epoch) + '_weight.pkl')
        return best_psnr, best_ssim, best_epoch


    def run(self):
        best_psnr = 0
        best_ssim = 0
        best_epoch = 0
        self.build_model()
        self.writer = SummaryWriter(self.model_out_path + '/tensorboard')
        self.writer.add_graph(self.netG, input_to_model=torch.randn(1, 3, 32, 32).to('cuda:0'), verbose=False)

        self.netG.load_state_dict(torch.load(self.weight, map_location='cuda:0')['weight']) 
        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)
        self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)
        
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Running Epoch {} starts".format(epoch))

            self.train(epoch)
            temp_psnr, temp_ssim = self.test(epoch)
            self.schedulerD.step()
            self.schedulerG.step()

            if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                best_psnr = temp_psnr
                best_ssim = temp_ssim
                best_epoch = epoch
                self.save(best_psnr, best_ssim, best_epoch)
        
        self.save(best_psnr, best_ssim, self.nEpochs)
            
        return best_psnr, best_ssim, best_epoch
    

    def resume(self):
        best_psnr = 0
        best_ssim = 0
        best_epoch = 0
        self.build_model()
        self.writer = SummaryWriter(self.model_out_path + '/tensorboard')
        self.writer.add_graph(self.netG, input_to_model=torch.randn(1, 3, 32, 32).to('cuda:0'), verbose=False)

        checkpoint = torch.load(self.checkpoint, map_location='cuda:0')
        self.netG.load_state_dict(checkpoint['G_state_dict']) 
        self.netD.load_state_dict(checkpoint['D_state_dict']) 
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        start_epoch = checkpoint['epoch'] 
        best_epoch = checkpoint['epoch'] 
        self.optimizerG.load_state_dict(checkpoint['optimizeG_state_dict'])  
        self.optimizerD.load_state_dict(checkpoint['optimizeD_state_dict']) 

        self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])  
        self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict']) 
        # self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5, last_epoch = start_epoch-1)
        # self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5, last_epoch = start_epoch-1)

        for epoch in range(start_epoch + 1, start_epoch + self.nEpochs + 1):
            print("\n===> Resuming Epoch {} starts".format(epoch))

            self.train(epoch)
            temp_psnr, temp_ssim = self.test(epoch)
            self.schedulerD.step()
            self.schedulerG.step()

            if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                best_psnr = temp_psnr
                best_ssim = temp_ssim
                best_epoch = epoch
                self.save(best_psnr, best_ssim, best_epoch)
        
        self.save(best_psnr, best_ssim, start_epoch+self.nEpochs)

        return best_psnr, best_ssim, best_epoch

