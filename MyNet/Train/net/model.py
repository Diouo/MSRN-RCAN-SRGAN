import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.vgg import vgg19
from torchvision.models.feature_extraction import create_feature_extractor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=5 // 2)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=5 // 2)
        self.confusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=stride)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.avg_conv2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        y1 = self.prelu(self.conv1(x))
        y1 = self.conv2(y1)

        y2 = self.prelu(self.conv3(x))
        y2 = self.conv4(y2)

        y = torch.cat((y1, y2), dim=1)
        y = self.confusion(y)

        z = self.avg_pool(y)
        z = self.avg_conv1(z)
        z = self.relu(z)
        z = self.avg_conv2(z)
        z = self.sigmoid(z)

        out = z * y + x
        return  out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channel=3, base_filter=64):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), ResidualBlock(in_channels=base_filter, out_channels=base_filter, kernel=3, stride=1))

        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)

        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(base_filter))

        self.conv3 = nn.Conv2d(base_filter, num_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.prelu(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)

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
        self.bn2 = nn.BatchNorm2d(base_filter)
        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 2)
        self.conv4 = nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filter * 2)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.conv6 = nn.Conv2d(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(base_filter * 4)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filter * 8)
        self.conv8 = nn.Conv2d(base_filter * 8, base_filter * 8, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filter * 8)

        # Replaced original paper FC layers with FCN
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features = 512, out_features = 1024)
        self.linear2 = nn.Linear(in_features = 1024, out_features = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))

        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.leakyrelu(self.bn4(self.conv4(x)))
        x = self.leakyrelu(self.bn5(self.conv5(x)))
        x = self.leakyrelu(self.bn6(self.conv6(x)))
        x = self.leakyrelu(self.bn7(self.conv7(x)))
        x = self.leakyrelu(self.bn8(self.conv8(x)))

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
