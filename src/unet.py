import torch
import torch.nn as tnn

class UNetDown(tnn.Module):
    '''Utility class describing a decrease in spatial height and width from the
    previous layer, followed by convolution and activation'''
    def __init__(self, in_ch, out_ch, double=False):
        super(UNetDown, self).__init__()
        self.double = double
        
        self.net1 = tnn.Sequential(
            tnn.MaxPool2d(2),
            tnn.Conv2d(in_ch, out_ch, 3, 1, 1),
            tnn.BatchNorm2d(out_ch),
            tnn.ReLU(inplace=True)
        )
        
        if double:
            self.net2 = tnn.Sequential(
                tnn.Conv2d(out_ch, out_ch, 3, 1, 1),
                tnn.BatchNorm2d(out_ch),
                tnn.ReLU(inplace=True)
            )
            
    def forward(self, x):
        ret = self.net1(x)
        
        if self.double:
            ret = self.net2(ret)
            
        return ret
    
class UNetUp(tnn.Module):
    '''Utility class describing an increase in spatial height and width from the
    previous layer, followed by convolution and activation'''
    def __init__(self, in_ch, out_ch, double=False):
        super(UNetUp, self).__init__()
        self.double = double
        
        self.up = tnn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.net1 = tnn.Sequential(
                tnn.Conv2d(in_ch, out_ch, 3, 1, 1),
                tnn.BatchNorm2d(out_ch),
                tnn.ReLU(inplace=True)
            )
        
        if self.double:
            self.net2 = tnn.Sequential(
                tnn.Conv2d(out_ch, out_ch, 3, 1, 1),
                tnn.BatchNorm2d(out_ch),
                tnn.ReLU(inplace=True)
            )
        
    def forward(self, l1, l0):
        l1 = self.up(l1)
        ret = torch.cat([l1, l0], dim=1)
        ret = self.net1(ret)
        if self.double:
            ret = self.net2(ret)
            
        return ret

class UNetLite(tnn.Module):
    '''A version of U-Net with parameters that can either scale the number of
    channels at each layer, or use single convolutions instead of double
    convolutions at each filter scale
    
    Args:
        in_ch : The number of input channels
        out_ch : The number of output channels
        scale : Scales the number of channels in each layer, with 1.0 being 
                equivalent to a standard U-Net implementation
        double : Set to True to perform two convolutions at every scale
    '''
    def __init__(self, in_ch=1, out_ch=1, scale=0.75, double=False):
        super(UNetLite, self).__init__()

        ch1 = int(64 * scale)
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        ch4 = ch3 * 2
        ch5 = ch4 * 2
        
        self.conv1 = tnn.Sequential(
            tnn.Conv2d(1, ch1, 3, 1, 1),
            tnn.BatchNorm2d(ch1),
            tnn.ReLU(inplace=True),
        )

        self.down2 = UNetDown(ch1, ch2, double)
        self.down3 = UNetDown(ch2, ch3, double)
        self.down4 = UNetDown(ch3, ch4, double)
        self.down5 = UNetDown(ch4, ch5, double)
        
        self.up4 = UNetUp(ch5, ch4, double)
        self.up3 = UNetUp(ch4, ch3, double)
        self.up2 = UNetUp(ch3, ch2, double)
        self.up1 = UNetUp(ch2, ch1, double)
        
        self.conv2 = tnn.Sequential(
            tnn.Conv2d(ch1, ch1, 3, 1, 1),
            tnn.BatchNorm2d(ch1),
            tnn.ReLU(inplace=True),
        )
        self.conv3 = tnn.Sequential(
            tnn.Conv2d(ch1, ch1, 3, 1, 1),
            tnn.BatchNorm2d(ch1),
            tnn.ReLU(inplace=True),
        )
        self.conv4 = tnn.Conv2d(ch1, 2, 3, 1, 1)
        
    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.down2(l1)
        l3 = self.down3(l2)
        l4 = self.down4(l3)
        l5 = self.down5(l4)
        u4 = self.up4(l5, l4)
        u3 = self.up3(u4, l3)
        u2 = self.up2(u3, l2)
        u1 = self.up1(u2, l1)
        
        c1 = self.conv2(u1)
        c2 = self.conv3(c1)
        c3 = self.conv4(c2)
        return c3
    