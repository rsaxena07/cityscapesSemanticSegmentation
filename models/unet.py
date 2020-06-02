import torch
import torch.nn as nn

def doubleConv(in_channels, out_channels, kernels=3, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernels, padding=pad),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernels, padding=pad),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    
    def __init__(self, input_channels, n_classes):
        super().__init__()
        
        self.downConv1 = doubleConv(input_channels, 64)
        self.downConv2 = doubleConv(64, 128)
        self.downConv3 = doubleConv(128, 256)
        self.downConv4 = doubleConv(256, 512)
        
        self.maxPoolLayer = nn.MaxPool2d(2) # kernel size will reduce both dimension by half
        self.upsampleLayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upConv3 = doubleConv(256+512, 256)
        self.upConv2 = doubleConv(128+256, 128)
        self.upConv1 = doubleConv(64+128, 64)
        
        self.finalConv = doubleConv(64, n_classes, kernels=1, pad=0) # final 1x1 kernel conv layer with depth=n_classes
        
    def forward(self, x):
        
        conv_r1 = self.downConv1(x)      #Conv Layer 1,2
        x = self.maxPoolLayer(conv_r1)   #Pool Layer 3
        
        conv_r2 = self.downConv2(x)      #Conv Layer 4,5
        x = self.maxPoolLayer(conv_r2)   #Pool Layer 6
        
        conv_r3 = self.downConv3(x)      #Conv Layer 7,8
        x = self.maxPoolLayer(conv_r3)   #Conv Layer 9
        
        x = self.downConv4(x)            #Conv Layer 10,11
        
        x = self.upsampleLayer(x)        #Upsampling Layer 12
        
        x = torch.cat([conv_r3, x], dim=1) # concatenate
        x = self.upConv3(x)              # Conv Layer 13,14
        
        x = self.upsampleLayer(x)        # Upsampling Layer 15
        
        x = torch.cat([conv_r2, x], dim=1) 
        x = self.upConv2(x)              # Conv Layer 16,17
        
        x = self.upsampleLayer(x)        # Upsampling Layer 18
        
        x = torch.cat([conv_r1, x], dim=1) 
        x = self.upConv1(x)              # Conv Layer 19,20
        
        out = self.finalConv(x)
        return out
    
    
class CityscapesUnet(nn.Module): #Wrapper Unet class for dataset classes & softmax
    
    def __init__(self, input_channels):
        super.__init__()
        
        n_classes=34          #Fixed for cityscapes dataset
        self.Unet = UNet(input_channels, 34)
        self.softmaxLayer = nn.Softmax(dim=1)  # softmax on depth of tensor
        
    def forward(self, X):
        
        X = self.softmaxLayer(self.Unet(X))
        return X