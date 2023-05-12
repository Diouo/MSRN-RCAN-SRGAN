import os
import numpy as np
import argparse
from math import log10
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("/home/guozy/BISHE/MyNet_ddp/code")
from dataset import get_training_set, get_test_set
from net.model_ddp import Generator, Discriminator, VGG19, RGB2Y


class MyNetTrainer(object):
    def __init__(self, config, model_out_path):
        super(MyNetTrainer, self).__init__()

        self.train_crop_size = config.train_crop_size
        self.train_dataset = config.train_dataset
        self.batchSize = config.batchSize
        self.test_crop_size = config.test_crop_size
        self.test_dataset = config.test_dataset
        self.testBatchSize = config.testBatchSize

        self.upscale_factor = config.upscale_factor
        self.nEpochs = config.nEpochs
        self.G_pretrain_epoch= config.G_pretrain_epoch
        self.num_residuals = config.num_residuals
        self.K = config.K
        self.G_lr = config.G_lr
        self.D_lr = config.D_lr

        self.netG = None
        self.netD = None
        self.criterionG = None
        self.criterionD = None
        self.criterionF= None
        self.optimizerG = None
        self.optimizerD = None
        self.schedulerG = None
        self.schedulerD = None
        self.feature_extractor = None
        self.RGB2Y = None
        self.training_loader = None
        self.testing_loader = None
        
        self.model_out_path = model_out_path
        self.checkpoint = config.checkpoint
        self.writer = SummaryWriter(self.model_out_path + '/tensorboard')

        # self.local_rank =  int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(self.local_rank)
        # dist.init_process_group(backend='nccl', world_size=2)
        # dist.barrier()
        # self.device = torch.device("cuda", self.local_rank)
        # world_size = torch.distributed.get_world_size()
        # if dist.get_rank() == 0:
        #     print('world size: ', world_size)

        torch.cuda.empty_cache()
        self.local_rank = config.local_rank
        self.device = torch.device("cuda", config.local_rank)
        dist.init_process_group(backend='nccl')
        # dist.barrier()
        self.world_size = torch.distributed.get_world_size()
        if dist.get_rank() == 0:
            print('\n===> world size: ', self.world_size)
        
        
    def build_model(self):
        print('\n===> Building the Model')

        # build Generator
        self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64, num_channel=3).to(self.local_rank)
        self.netG = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[self.local_rank], output_device=self.local_rank)
        self.criterionG = nn.MSELoss().to(self.local_rank)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.G_lr)
        
        # build Discriminator
        self.netD = Discriminator(base_filter=64, num_channel=3).to(self.local_rank)
        self.netD = torch.nn.parallel.DistributedDataParallel(self.netD, device_ids=[self.local_rank])
        self.criterionD = nn.BCELoss().to(self.local_rank)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.D_lr)

        # build feature extractor
        self.feature_extractor = VGG19().to(self.local_rank)
        self.feature_extractor.eval()

        # build RGB2Y
        self.rgb2y = RGB2Y().to(self.local_rank)
        self.rgb2y.load_state_dict(torch.load('/home/guozy/BISHE/OtherNet/RGB2Y.pkl'))
        for model_parameters in self.rgb2y.parameters():
            model_parameters.requires_grad = False
        self.rgb2y.eval()


    def get_dataset(self):
        print('\n===> Loading datasets')

        train_set = get_training_set(self.upscale_factor,self.train_crop_size, self.train_dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        test_set = get_test_set(self.upscale_factor, self.test_crop_size, self.test_dataset)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
        self.training_loader = DataLoader(dataset=train_set, batch_size=self.batchSize, num_workers=8, pin_memory=True, sampler=train_sampler)
        self.testing_loader = DataLoader(dataset=test_set, batch_size=self.testBatchSize, num_workers=4, pin_memory=True, shuffle=False)


    def G_pretrain(self,epoch):
        print('     G Pretraining')

        self.netG.train()
        self.netD.eval()

        g_loss = 0 # only mse loss
        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            data = data.to(self.local_rank)
            target = target.to(self.local_rank)

            self.optimizerG.zero_grad()
            g_real = self.netG(data) # fake samples, torch.Size([4, 3, 1024, 1024])
            mse_loss = self.criterionG(g_real, target) # MSE loss of fake samples
            mse_loss.backward()
            self.optimizerG.step()
            dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
            g_loss +=  mse_loss / self.world_size

        if self.local_rank == 0:
            self.writer.add_scalar(tag="train/G_loss", scalar_value=g_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/G_lr", scalar_value=self.optimizerG.state_dict()['param_groups'][0]['lr'], global_step=epoch)


    def G_train(self, epoch):
        print('     G Training')

        self.netG.train()
        self.netD.eval()

        # ===========================================================
        # Train Generator
        # ===========================================================
        g_loss = 0
        g_mse_loss = 0
        g_gan_loss = 0
        g_content_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            self.optimizerG.zero_grad()

            # setup noise
            data = data.to(self.local_rank)
            target = target.to(self.local_rank)
            real_label = torch.ones(data.size(0), 1).to(self.local_rank)

            g_real = self.netG(data).to(self.local_rank) # fake samples, torch.Size([4, 3, 1024, 1024])
            g_fake = self.netD(g_real) # prob of fake samples
            mse_loss = self.criterionG(g_real, target) # MSE loss of fake samples
            gan_loss = self.criterionD(g_fake, real_label) # Adversarial loss of fake samples
            content_loss = self.feature_extractor.forward(g_real,target) # VGG loss of fake samples

            total_loss = mse_loss + 1e-3 * gan_loss + 0.006 * content_loss # total loss of G
            total_loss.backward()
            self.optimizerG.step()

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(gan_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(content_loss, op=dist.ReduceOp.SUM)
            g_loss += total_loss / self.world_size
            g_mse_loss += mse_loss / self.world_size
            g_gan_loss += gan_loss / self.world_size
            g_content_loss += content_loss / self.world_size

        if self.local_rank == 0:
            self.writer.add_scalar(tag="train/G_loss", scalar_value=g_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/G_mse_loss", scalar_value=g_mse_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/G_gan_loss", scalar_value=1e-3 * g_gan_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/G_content_loss", scalar_value=0.006 * g_content_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/G_lr", scalar_value=self.optimizerG.state_dict()['param_groups'][0]['lr'], global_step=epoch)


    def D_train(self,epoch):
        print('     D Training')
        
        self.netG.eval()
        self.netD.train()
        
        # ===========================================================
        # Train Discriminator
        # ===========================================================
        d_loss = 0
        d_real_total = 0
        d_fake_total = 0
        for batch_num, (data, target) in enumerate(self.training_loader): # torch.Size([4, 3, 64, 64]), torch.Size([4, 3, 256, 256])
            self.optimizerD.zero_grad()

            data = data.to(self.local_rank)
            target = target.to(self.local_rank)
            real_label = torch.ones(data.size(0), 1).to(self.local_rank)
            fake_label = torch.zeros(data.size(0), 1).to(self.local_rank)

            d_real = self.netD(target) # prob of real samples
            d_real_loss = self.criterionD(d_real, real_label) # BCE loss of real samples
            temp = d_real_loss.clone()
            dist.all_reduce(temp, op=dist.ReduceOp.SUM)
            if (temp / self.world_size) > 0.4:
                d_real_loss.backward()

            d_fake = self.netD(self.netG(data)) # prob of fake samples
            d_fake_loss = self.criterionD(d_fake, fake_label) # BCE loss of fake samples
            temp = d_fake_loss.clone()
            dist.all_reduce(temp, op=dist.ReduceOp.SUM)
            if (temp / self.world_size) > 0.4:
                d_fake_loss.backward()

            d_total =  d_real_loss + d_fake_loss  # total loss of D
            self.optimizerD.step()

            dist.all_reduce(d_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(d_real_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(d_fake_loss, op=dist.ReduceOp.SUM)
            d_loss += d_total / self.world_size
            d_real_total += d_real_loss / self.world_size
            d_fake_total += d_fake_loss / self.world_size
            
        if self.local_rank == 0:
            self.writer.add_scalar(tag="train/D_loss", scalar_value=d_loss / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/D_real_loss", scalar_value=d_real_total / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/D_fake_loss", scalar_value=d_fake_total / (batch_num + 1), global_step=epoch)
            self.writer.add_scalar(tag="train/D_lr", scalar_value=self.optimizerD.state_dict()['param_groups'][0]['lr'], global_step=epoch)


    def test(self,epoch):
        print('     Testing')

        self.netG.eval()
        self.netD.eval()

        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.testing_loader):
                data = data.to('cuda:0'),
                target = target.to('cuda:0')
                prediction = self.netG(data[0]).clamp(0,1)
                mse = self.criterionG(prediction, target)
                avg_psnr += 10 * log10(1 / mse.item())
                avg_ssim += ssim(prediction.squeeze(dim=0).cpu().numpy(), target.squeeze(dim=0).cpu().numpy(), channel_axis=0) 
        
        img = Image.open('/home/guozy/BISHE/dataset/Set5/butterfly.png')
        data = (ToTensor()(img)) 
        data = data.to('cuda:0').unsqueeze(0) # torch.Size([1, 3, 256, 256])
        out = self.netG(data).detach().squeeze(0).clamp(0,1) # torch.Size([3, 1024, 1024])

        print('     psnr:{}, ssim:{}'.format(avg_psnr/ len(self.testing_loader), avg_ssim/ len(self.testing_loader)))
        self.writer.add_scalar(tag="test/PSNR", scalar_value=avg_psnr / len(self.testing_loader), global_step=epoch)
        self.writer.add_scalar(tag="test/SSIM", scalar_value=avg_ssim / len(self.testing_loader), global_step=epoch)
        self.writer.add_image("test/IMAGE", out, epoch, dataformats='CHW')

        return avg_psnr, avg_ssim


    def test_Y(self,epoch):
        print('     Testing')

        self.netG.eval()
        self.netD.eval()

        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.testing_loader):
                data = data.to(self.local_rank),
                target = target.to(self.local_rank).mul(255.0)
                prediction = self.netG(data[0]).clamp(0,1).mul(255.0)

                target = self.rgb2y(target)
                prediction = self.rgb2y(prediction)
                mse = self.criterionG(target, prediction)

                avg_psnr += 10 * log10(255 * 255 / mse)
                avg_ssim += ssim(prediction.squeeze(dim=0).cpu().numpy().astype(np.uint8), target.squeeze(dim=0).cpu().numpy().astype(np.uint8), channel_axis=0) 
        
        img = Image.open('/home/guozy/BISHE/dataset/Set5/butterfly.png')
        data = (ToTensor()(img)) 
        data = data.to(self.local_rank).unsqueeze(0) # torch.Size([1, 3, 256, 256])
        out = self.netG(data).detach().squeeze(0).clamp(0,1) # torch.Size([3, 1024, 1024])

        print('     psnr:{}, ssim:{}'.format(avg_psnr/ len(self.testing_loader), avg_ssim/ len(self.testing_loader)))
        self.writer.add_scalar(tag="test/PSNR", scalar_value=avg_psnr / len(self.testing_loader), global_step=epoch)
        self.writer.add_scalar(tag="test/SSIM", scalar_value=avg_ssim / len(self.testing_loader), global_step=epoch)
        self.writer.add_image("test/IMAGE", out, epoch, dataformats='CHW')

        return avg_psnr, avg_ssim


    def save(self, best_psnr, best_ssim, epoch):
        print('     Saving')
        checkpoint={
            'epoch':epoch,
            'D_state_dict':self.netD.state_dict(),
            'G_state_dict':self.netG.state_dict(),
            'optimizeG_state_dict':self.optimizerG.state_dict(),
            'optimizeD_state_dict':self.optimizerD.state_dict(),
            # new add, not in baseline
            # 'schedulerG_state_dict':self.schedulerG.state_dict(),
            # 'schedulerD_state_dict':self.schedulerD.state_dict(),
            'best_psnr':best_psnr,
            'best_ssim':best_ssim,
                    }
        checkpoints_out_path = self.model_out_path +'/checkpoints/'
        torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')
    
    
    def pretrain(self):
        self.build_model()
        self.get_dataset()
        checkpoints_out_path = self.model_out_path +'/checkpoints/'

        self.optimizerG = optim.Adam([{'params': filter(lambda p: p.requires_grad, self.netG.parameters()), 'initial_lr': self.G_lr}], lr=self.G_lr)
        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[500,1000,1500], gamma=0.5)

        best_psnr = 0
        best_ssim = 0
        best_epoch = 0   
        dist.barrier()
        for epoch in range(1, self.G_pretrain_epoch + 1):

            self.training_loader.sampler.set_epoch(epoch)
            if self.local_rank == 0:
                print('\n===> G Pretraining Epoch {} starts'.format(epoch))
            dist.barrier()

            self.G_pretrain(epoch)
            self.schedulerG.step()

            if self.local_rank == 0:
                temp_psnr, temp_ssim = self.test_Y(epoch)

                if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                    best_psnr = temp_psnr
                    best_ssim = temp_ssim
                    best_epoch = epoch

                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

                elif epoch % 50 == 0:
                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

                elif epoch == self.G_pretrain_epoch:
                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

        return best_psnr, best_ssim, best_epoch


    def pretrain_resume(self):
        self.build_model()
        self.get_dataset()
        checkpoint = torch.load(self.checkpoint, map_location='cuda:{}'.format(self.local_rank))
        checkpoints_out_path = self.model_out_path +'/checkpoints/'

        weights_dict = {}
        for k, v in checkpoint['G_state_dict'].items():
            new_k = 'module.' + k
            weights_dict[new_k] = v
        self.netG.load_state_dict(weights_dict)
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        start_epoch = checkpoint['epoch'] 
        best_epoch = checkpoint['epoch'] 

        self.optimizerG = optim.Adam([{'params': filter(lambda p: p.requires_grad, self.netG.parameters()), 'initial_lr': self.G_lr}], lr=self.G_lr)
        # self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[402,800,1200,1600], gamma=0.5, last_epoch=start_epoch)

        dist.barrier()
        for epoch in range(start_epoch + 1, start_epoch + 1 + self.G_pretrain_epoch + 1):
            self.training_loader.sampler.set_epoch(epoch)
            # self.testing_loader.sampler.set_epoch(epoch)
            if self.local_rank == 0:
                print('\n===> G Pretraining Epoch {} starts'.format(epoch))
            dist.barrier()

            self.G_pretrain(epoch)
            # self.schedulerG.step()
            
            if self.local_rank == 0:
                temp_psnr, temp_ssim = self.test_Y(epoch)

                if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                    best_psnr = temp_psnr
                    best_ssim = temp_ssim
                    best_epoch = epoch

                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

                elif epoch % 50 == 0:
                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

                elif epoch == start_epoch + 1 + self.G_pretrain_epoch:
                    print('     Saving')
                    checkpoint = {'G_state_dict':self.netG.module.state_dict(), 'epoch':epoch,'best_psnr':best_psnr,'best_ssim':best_ssim}
                    torch.save(checkpoint, checkpoints_out_path + str(epoch) + '_checkpoint.pkl')

        return best_psnr, best_ssim, best_epoch


    def run(self):
        self.build_model()
        self.get_dataset()
        checkpoint = torch.load(self.checkpoint, map_location='cuda:{}'.format(self.local_rank))

        weights_dict = {}
        for k, v in checkpoint['G_state_dict'].items():
            new_k = 'module.' + k
            weights_dict[new_k] = v
        self.netG.load_state_dict(weights_dict)

        # self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)
        # self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5)

        best_psnr = 0
        best_ssim = 0
        best_epoch = 0
        dist.barrier()
        for epoch in range(1, self.nEpochs + 1):
            dist.barrier()
            if self.local_rank == 0:
                print("\n===> Running Epoch {} starts".format(epoch))
            dist.barrier()
            if (epoch-1) % self.K == 0:
                self.D_train(epoch)
            dist.barrier()
            self.G_train(epoch)
            # self.schedulerD.step()
            # self.schedulerG.step()

            if self.local_rank == 0:
                temp_psnr, temp_ssim = self.test_Y(epoch)

                if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                    best_psnr = temp_psnr
                    best_ssim = temp_ssim
                    best_epoch = epoch
                    self.save(best_psnr, best_ssim, epoch)

                elif epoch % 50 == 0:
                    self.save(best_psnr, best_ssim, epoch)

                elif epoch == self.nEpochs:
                    self.save(best_psnr, best_ssim, epoch)
            
        return best_psnr, best_ssim, best_epoch
    

    def run_resume(self):
        self.build_model()
        self.get_dataset()
        checkpoint = torch.load(self.checkpoint, map_location='cuda:{}'.format(self.local_rank))

        # weights_dict = {}
        # for k, v in checkpoint['G_state_dict'].items():
        #     new_k =  k[7:]
        #     weights_dict[new_k] = v
        # self.netG.load_state_dict(weights_dict)
        self.netG.load_state_dict(checkpoint['G_state_dict'])

        # weights_dict = {}
        # for k, v in checkpoint['D_state_dict'].items():
        #     new_k = k[7:]
        #     weights_dict[new_k] = v
        # self.netD.load_state_dict(checkpoint['D_state_dict'])
        self.netD.load_state_dict(checkpoint['D_state_dict'])



        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        start_epoch = checkpoint['epoch'] 
        best_epoch = checkpoint['epoch'] 
        self.optimizerG.load_state_dict(checkpoint['optimizeG_state_dict'])  
        self.optimizerD.load_state_dict(checkpoint['optimizeD_state_dict']) 

        # self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])  
        # self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict']) 
        # self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5, last_epoch = start_epoch-1)
        # self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 100, 150, 200, 300, 350], gamma=0.5, last_epoch = start_epoch-1)

        dist.barrier()
        for epoch in range(start_epoch + 1, start_epoch + 1 + self.nEpochs + 1):
            dist.barrier()
            if self.local_rank == 0:
                print("\n=== >Persuming Running Epoch {} starts".format(epoch))
            dist.barrier()
            # if (epoch-1) % self.K == 0:
            for i in range(self.K):
                self.D_train(epoch)
            dist.barrier()
            self.G_train(epoch)
            # self.schedulerD.step()
            # self.schedulerG.step()

            if self.local_rank == 0:
                temp_psnr, temp_ssim = self.test_Y(epoch)

                if temp_psnr >= best_psnr and temp_ssim >= best_ssim:
                    best_psnr = temp_psnr
                    best_ssim = temp_ssim
                    best_epoch = epoch
                    self.save(best_psnr, best_ssim, epoch)

                elif epoch % 50 == 0:
                    self.save(best_psnr, best_ssim, epoch)

                elif epoch == self.nEpochs:
                    self.save(best_psnr, best_ssim, epoch)
            
        return best_psnr, best_ssim, best_epoch

