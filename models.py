import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class NetBN(nn.Module):
    def __init__(self):
        super(NetBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv10 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.gap = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv11 = nn.Conv2d(in_channels=4096, out_channels=10, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.pool1(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
        )
        x = self.pool2(
            F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x))))))))
        )
        x = self.gap(
            F.relu(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x))))))
        )
        x = self.conv11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
    
    def summary(self, input_size):
        print(summary(self, input_size))
