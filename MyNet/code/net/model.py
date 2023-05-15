import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.vgg import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=True)
        self.conv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=True)

        self.conv5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=5 // 2, bias=True)
        self.conv5_2 = nn.Conv2d(out_channels , out_channels , kernel_size=5, stride=stride, padding=5 // 2, bias=True)

        self.confusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=stride)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.avg_conv2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y1 = self.leakyrelu(self.conv3_1(x))
        y1 = self.conv3_2(y1) + x

        y2 = self.leakyrelu(self.conv5_1(x))
        y2 = self.conv5_2(y2) + x

        y = torch.cat((y1, y2), dim=1)
        y = self.confusion(y) 

        z = self.avg_pool(y)
        z = self.avg_conv1(z)
        z = self.leakyrelu(z)
        z = self.avg_conv2(z)
        z = self.sigmoid(z)

        out = z * y +  x
        return  out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        # self.prelu = nn.PReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        # self.shuffler = nn.PixelShuffle(4)
        self.upsample =  nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        x = self.leakyrelu(self.conv1(self.upsample(x)))
        x = self.leakyrelu(self.conv2(self.upsample(x)))
        return x


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor=4, num_channel=3, base_filter=64):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # define head module
        self.head = nn.ModuleList()
        self.head.append(nn.Conv2d(num_channel, base_filter, kernel_size=3, stride=1, padding=1, bias=True))
        self.head = nn.Sequential(*self.head)

        # define body module
        self.body = nn.ModuleList()
        for _ in range(n_residual_blocks):
            self.body.append(ResidualBlock(in_channels=base_filter, out_channels=base_filter, kernel=3, stride=1))
        self.body = nn.Sequential(*self.body)

        # define tail module
        self.tail = nn.ModuleList()
        self.tail.append(nn.Conv2d(base_filter * (n_residual_blocks + 1) + 3, base_filter, kernel_size=3, stride=1, padding=1, bias=True))
        self.tail.append(UpsampleBlock(base_filter))
        self.tail.append(nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True))
        self.tail.append(nn.LeakyReLU(negative_slope=0.2))
        self.tail.append(nn.Conv2d(base_filter, num_channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.tail = nn.Sequential(*self.tail)

        
    def forward(self, x):
        concat = [x]
        bicubic = self.upsample(x)

        x = self.head(x)
        concat.append(x)

        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
            concat.append(x)
        
        x = torch.cat(concat,1)
        x = self.tail(x)

        return x + bicubic

    def weight_init(self, mean=0.0, std=0.02) -> None:
        for m in self._modules:
            module = self._modules[m]
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
    

class Discriminator(nn.Module):
    def __init__(self, num_channel=3, base_filter=64):
        super(Discriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(base_filter * 8, base_filter * 8, kernel_size=3, stride=2, padding=1)

        # Replaced original paper FC layers with FCN
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features = 512, out_features = 1024)
        self.linear2 = nn.Linear(in_features = 1024, out_features = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.leakyrelu(self.conv5(x))
        x = self.leakyrelu(self.conv6(x))
        x = self.leakyrelu(self.conv7(x))
        x = self.leakyrelu(self.conv8(x))

        x = self.avepool(x)
        x = self.flatten(x)
        x = self.leakyrelu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        
        return x

    def weight_init(self, mean=0.0, std=0.02) -> None:
        for m in self._modules:
            module = self._modules[m]
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

class VGG19(nn.Module):
    def __init__(
            self,
            feature_model_extractor_node = "features.35",
            feature_model_normalize_mean = [0.485, 0.456, 0.406],
            feature_model_normalize_std= [0.229, 0.224, 0.225]
    ) -> None:
        super(VGG19, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = vgg19(weights='IMAGENET1K_V1')
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. 
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor, gt_tensor):
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F.mse_loss(sr_feature, gt_feature)

        return loss


class RGB2Y(nn.Module):
    def __init__(self):
        super(RGB2Y, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1,bias=True)
        weight = torch.Tensor([0.2570,0.5040,0.0980]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        bias = torch.Tensor([16.0])

        conv_dict = {'weight':weight, 'bias':bias}
        self.conv.load_state_dict(conv_dict)

    def forward(self, x):
        return self.conv(x)

 
class SSIM(torch.nn.Module):
    def __init__(self, window_size=7,L=255.0):
        super(SSIM, self).__init__()
        self.window = torch.ones((window_size,window_size)) / (window_size * window_size)
        self.window = self.window.unsqueeze(0).unsqueeze(0)
        self.L = L
        self.pad = 0
        self.channel = 1

    def forward(self, img1, img2):

            mu1 = F.conv2d(img1, self.window, padding=self.pad, groups=self.channel)
            mu2 = F.conv2d(img2, self.window, padding=self.pad, groups=self.channel)
        
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
        
            sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.pad, groups=self.channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.pad, groups=self.channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, self.window, padding=self.pad, groups=self.channel) - mu1_mu2
        
            C1 = (0.01 * self.L) ** 2
            C2 = (0.03 * self.L) ** 2
        
            v1 = 2.0 * sigma12 + C2
            v2 = sigma1_sq + sigma2_sq + C2
        
            return torch.mean(((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2))
