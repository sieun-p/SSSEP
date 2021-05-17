import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    
    def __init__(self):
        super(EEGNet, self).__init__()
    
        # Conv2D Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64))
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Depthwise Layer
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(8, 1),
                                   groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        
        # Separable Layer
        self.separable = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d(1, 8)

        #Flatten
        self.flatten = nn.Flatten()
        
        #linear
        self.linear1 = nn.Linear(96,4)
        

    def forward(self, x):

        print("연산 전", x.size())
        # Conv2D
        x = F.pad(x,(31,32,0,0))
        x = self.conv1(x)
        print("conv1", x.size())
        x = self.batchnorm1(x)    
        print("batchnorm", x.size())

        # Depthwise conv2D
        x = self.depthwise(x)
        print("depthwise", x.size())
        x = F.elu(self.batchnorm2(x))
        print("batchnorm & elu", x.size())
        x = self.pooling1(x)
        print("pooling", x.size())
        x = F.dropout(x, 0.5)
        print("dropout", x.size())
        
        # Separable conv2D
        x = self.separable(x)
        print("separable", x.size())
        x = F.elu(self.batchnorm3(x))
        print("batchnorm & elu", x.size())
        x = self.pooling2(x)
        print("pooling", x.size())
        x = F.dropout(x, 0.5)
        print("dropout", x.size())
        
        #Flatten
        x = self.flatten(x)
        print("flatten", x.size())
        
        # FC Layer
        x = F.softmax(self.linear1(x))
        print("linear", x.size())
        
        return x
    
#model = EEGNet()
#myModel = model(torch.randn(10,1,64,128))