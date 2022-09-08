# %%
import re
import torch as ts 
import torchvision as tv 
import numpy as np 
import matplotlib as plt 
import antialiased_cnns
import matplotlib.pyplot as plt
import torch.nn as nn
#import keras.utils
from PIL import Image 
import numpy as np
import sys

import torch
import torchvision
import numpy as np
from torch import nn
from torch import autograd 
import skimage as measure



# %%
class Down(nn.Module):
    def __init__(self, intake : int  ,out : int , stride :int ,leaky_relu : np.float64, kenrel = 3 ) -> None:
        super(Down, self).__init__()
        # Ein U Net in jeder Layer besteht immer aus 2 Convs und einer Pooling operatation | Downwards 
        self.seq = nn.Sequential( 
            nn.Conv2d(in_channels= intake ,out_channels= out, kernel_size= 3 , padding= 1),
            nn.BatchNorm2d(out), 
            nn.LeakyReLU(leaky_relu),
            nn.Conv2d(in_channels= out ,out_channels= out, kernel_size= 3 ,padding= 1),
            nn.BatchNorm2d(out), 
            antialiased_cnns.BlurPool(out, stride= 2, filt_size= 3), 
            nn.LeakyReLU(leaky_relu))
    def forward(self,x):
        x = self.seq(x)
        return x

class Up(nn.Module):
    def __init__(self, intake :int , out :int ) -> None:
        super(Up,self).__init__()
        self.seq = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor= 2),
            nn.Conv2d(in_channels= intake ,out_channels= out, kernel_size= 3 , padding=1),
            nn.Conv2d(in_channels= out ,out_channels= out, kernel_size= 3 , padding=1),
            )
    def forward(self,x):
        x = self.seq(x)
        return x 

# %%
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net,self).__init__()
        self.encoder_layers = [
            Down(3,64,2,0.01),
            Down(64,128,2,0.01),
            Down(128,256,2,0.01),
            Down(256,512,2,0.01)
        ]
        self.decoder_layers = [
            Up(512 + 512 ,256),
            Up(256 + 256 ,256),
            Up(128 +256  ,256),
            Up(64 + 256 ,256)

        ]

    def forward(self,x):
        skip = []
        for layer in self.encoder_layers: 
            x = layer(x)
            skip.append(x)
        for count,layer in enumerate(self.decoder_layers):
            #print(count, ts.cat((x,skip[len(self.decoder_layers)- 1  - count ]).shape))
            x = layer(ts.cat((x,skip[len(self.decoder_layers)- 1  - count ]),dim= 1))
        return x 
    def middle(self,x): 
        # illumination code is returned with this 
        for layer in self.encoder_layers: 
            x = layer(x)
            return x 



# %%
class F_Net(nn.Module):
    def __init__(self):
        super(F_Net,self).__init__()
        self.layers =[nn.Sequential(nn.Linear(512,512), nn.SiLU()) for x in range(8)]
        self.ouput = nn.Sequential(nn.Linear(512,4))
        self.relu = nn.ReLU()
    
    def forward(self,x,x_feature): 
        skip = torch.cat((x,x_feature),dim = 0)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x +=  skip
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[7](x)
        x = self.ouput(x)
        col = self.relu(x[1:3])
        return x[0], col

class Shading(nn.Module):
    def __init__(self):
        super(Shading,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(16+3,256),nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(16+3,256),nn.SiLU())
        self.layer3 = nn.Sequential(nn.Linear(16+3,256),nn.SiLU())
        self.out = nn.ReLU()
    def forward(self,normal, l):
        x = torch.cat((l,normal),dim = 0)
        x =  self.layer1(x)
        x =  self.layer2(x)
        x =  self.layer3(x)
        x = self.out(x)
        return x


        
        

    

