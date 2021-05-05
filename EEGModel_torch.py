import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.module):
    def __init__(self):
        super(EEGNet, self).__init__()
    
        # Conv2D Layer
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding = "same")
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Depthwise Layer
        self.depthwise = nn.Conv2d(8, 16, (8, 1), padding ="valid", groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        
        # Separable Layer
        self.separable = nn.Conv2d(16, 16, (1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d((1, 8))

        # FC Layer
        self.fc1 =
        

    def forward(self, x):

        # Conv2D
        x = self.conv1(x)
        x = self.batchnorm1(x)        

        # Depthwise
        x = self.depthwise(x)
        x = F.elu(self.batchnorm2(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)
        
        # Separable
        x = self.separable(x)
        x = F.elu(self.batchnorm3(x))
        x = self.pooling2(x)
        x = F.dropout(x, 0.5)
