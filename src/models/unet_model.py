""" Full assembly of the parts to form the complete network """
import os
import sys

from src.models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,aux=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.aux=aux
        if aux is True:
            self.aux1_head=Aux_Head(64,n_classes)
            self.aux2_head=Aux_Head(128,n_classes)
            self.aux3_head=Aux_Head(256,n_classes)
            self.aux4_head=Aux_Head(512,n_classes)
            self.aux5_head=Aux_Head(512,n_classes)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        _,_,h,w=x.size()
        feature={}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        feature['x1'] = x1
        feature['x2'] = x2
        feature['x3'] = x3
        feature['x4'] = x4
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feature['u2'] = x
        x = self.up3(x, x2)
        feature['u3'] = x
        x = self.up4(x, x1)
        feature['u4'] = x
        logits = self.outc(x)
        feature['out']=logits
        if self.aux is True:
            aux_output1 = self.aux1_head(x1)
            aux_output2 = self.aux2_head(x2)
            aux_output3 = self.aux3_head(x3)
            aux_output4 = self.aux4_head(x4)
            aux_output5 = self.aux5_head(x5)
            return {"output":logits,"aux1":aux_output1,"aux2":aux_output2,"aux3":aux_output3,"aux4":aux_output4,"aux5":aux_output5}
        return logits