import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device=torch.device("cpu")):
        super().__init__()
        
        encoder = self._build_encoder(in_channels)
        decoder = self._build_decoder(out_channels)
        self.model = nn.Sequential(encoder, decoder).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
    def forward(self, x):
        return self.model(x)

    def _build_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
    
    def _build_decoder(self, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=12, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    

class AutoEncoder2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device=torch.device("cpu")):
        super().__init__()
        
        encoder = self._build_encoder(in_channels)
        decoder = self._build_decoder(out_channels)
        self.model = nn.Sequential(encoder, decoder).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
    def forward(self, x):
        return self.model(x)

    def _build_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
    
    def _build_decoder(self, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
            

class AutoEncoder3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device=torch.device("cpu")):
        super().__init__()
        
        encoder = self._build_encoder(in_channels)
        decoder = self._build_decoder(out_channels)
        self.model = nn.Sequential(encoder, decoder).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
    def forward(self, x):
        return self.model(x)

    def _build_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
    
    def _build_decoder(self, out_channels):
        return nn.Sequential(
            #nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )