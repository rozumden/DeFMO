import torch
import torch.nn as nn
import torchvision.models
from main_settings import g_use_selfsupervised_timeconsistency, g_timeconsistency_type

class RenderingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.get_model_resnet()
        # self.net = self.get_model_resnet_smaller()
        self.rgba_operation = nn.Sigmoid()

    def get_model_resnet_smaller(self):
        model = nn.Sequential(
            nn.Conv2d(1025, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            torchvision.models.resnet.Bottleneck(1024,256),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(256,64),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(64,16),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        return model

    def get_model_resnet(self):
        last_channels = 4
        if g_use_selfsupervised_timeconsistency and g_timeconsistency_type == 'oflow':
            last_channels += 2
        model = nn.Sequential(
            nn.Conv2d(2049, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            torchvision.models.resnet.Bottleneck(1024,256),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(256,64),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(64,16),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, last_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        return model

    def get_model_cnn(self):
        model = nn.Sequential(
            nn.Conv2d(513, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        return model

    def forward(self, latent, times):
        renders = []
        shuffled_times = []
        for ki in range(times.shape[0]): shuffled_times.append(torch.randperm(times.shape[1]))
        shuffled_times = torch.stack(shuffled_times,1).contiguous().T
        for ki in range(times.shape[1]):
            t_tensor = times[range(times.shape[0]),shuffled_times[:,ki]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, latent.shape[2], latent.shape[3])
            latenti = torch.cat((t_tensor,latent),1)
            result = self.net(latenti)
            renders.append(result)
        renders = torch.stack(renders,1).contiguous()
        renders[:,:,:4] = self.rgba_operation(renders[:,:,:4])
        for ki in range(times.shape[0]): 
            renders[ki,shuffled_times[ki,:]] = renders[ki,:].clone()
        return renders

