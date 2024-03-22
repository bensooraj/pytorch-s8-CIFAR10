import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class NetBN(nn.Module):
    def __init__(self):
        super(NetBN, self).__init__()
        # Input Block
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        
        # Convolution Block 1
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )

        # Transition Block 1
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolution Block 2
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.convBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.convBlock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )

        # Transition Block 2
        self.convBlock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolution Block 3
        self.convBlock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.convBlock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.convBlock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.gap = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        # Output Block
        self.convBlock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0, bias=True)
        )
    
    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.pool1(x)

        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.convBlock6(x)

        x = self.convBlock7(x)
        x = self.pool2(x)

        x = self.convBlock8(x)
        x = self.convBlock9(x)
        x = self.convBlock10(x)

        x = self.gap(x)

        x = self.convBlock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
    
    def summary(self, input_size):
        print(summary(self, input_size))
