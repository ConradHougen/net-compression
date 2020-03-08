import torch
import torch.nn as nn

from SBP_utils import Conv2d_SBP, Linear_SBP, SBP_layer 

#I set these to equally weight the kl terms 
#from different layers, change if desired.
generic_kl_weights = {
    'five': [0.2,0.2,0.2,0.2,0.2],
    'eight':[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
}
cfg = [64, 192, 384, 256, 256, 4096, 4096]

class SBP_ConvBlock(nn.Module):
    def __init__(self,input_channel,out_channel,kernel_size=3,stride=2, padding=1):
        super(SBP_ConvBlock,self).__init__()

        self.conv1 = Conv2d_SBP(input_channel,out_channel,kernel_size=kernel_size, stride=stride,padding=padding)
        
        #self.sbp = SBP_layer(out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.rel = nn.ReLU(inplace=True)


    def forward(self,x):
        if self.training:

            x, kl  = self.conv1(x)
            #x, kl = self.sbp(x)
            x = self.rel(x)
            x = self.bn1(x)
            return x, kl
        else: 
            x = self.conv1(x)
            #x = self.sbp(x)
            x = self.rel(x)
            x = self.bn1(x)
        return x



class SBP_Block(nn.Module):
    def __init__(self,input_channel,out_channel,kernel_size=3,stride=2, padding=1):
        super(SBP_Block,self).__init__()

        self.conv1 = nn.Conv2d(input_channel,out_channel,kernel_size=kernel_size, stride=stride,padding=padding)
        
        self.sbp = SBP_layer(out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.rel = nn.ReLU(inplace=True)


    def forward(self,x):
        if self.training:

            x = self.conv1(x)
            x, kl = self.sbp(x)
            x = self.rel(x)
            x = self.bn1(x)
            return x, kl
        else: 
            x = self.conv1(x)
            x = self.sbp(x)
            x = self.rel(x)
            x = self.bn1(x)
        return x


class SBP_AlexNet(nn.Module):

    def __init__(self, cfg, classes=100,sbp_linear=False, conv=False,kl_weights= generic_kl_weights['five']):
        super(SBP_AlexNet, self).__init__()
        
        if conv: 
            
            self.block1 = SBP_ConvBlock(3,cfg[0])
            self.mp1 = nn.MaxPool2d(kernel_size=3,stride=2)

            self.block2 = SBP_ConvBlock(cfg[0],cfg[1],kernel_size=3,stride=1,padding=1)
            self.mp2= nn.MaxPool2d(kernel_size=3,stride=2)

            self.block3 = SBP_ConvBlock(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1)
            self.block4 = SBP_ConvBlock(cfg[2],cfg[3],kernel_size=3, stride=1, padding=1)
            self.block5 = SBP_ConvBlock(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1)
            
        else: 
            self.block1 = SBP_Block(3,cfg[0])
            self.mp1 = nn.MaxPool2d(kernel_size=3,stride=2)

            self.block2 = SBP_Block(cfg[0],cfg[1],kernel_size=3,stride=1,padding=1)
            self.mp2= nn.MaxPool2d(kernel_size=3,stride=2)

            self.block3 = SBP_Block(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1)
            self.block4 = SBP_Block(cfg[2],cfg[3],kernel_size=3, stride=1, padding=1)
            self.block5 = SBP_Block(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1)
            
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
       
        self.conv = conv
        self.cfg = cfg
        self.sbp_linear = sbp_linear
        if sbp_linear == False:
            
            self.kl_weights = generic_kl_weights['five']
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(cfg[4] * 1 * 1, cfg[5]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(cfg[5], cfg[6]),
                    nn.ReLU(inplace=True),
                    nn.Linear(cfg[6], classes)
            )
            

        else:
            self.kl_weights = generic_kl_weights['eight']
            
            self.dr1 = nn.Dropout()
            self.lsbp1 =  nn.Linear_SBP(cfg[4] * 1 * 1, cfg[5])
            self.rel2 = nn.ReLU(inplace=True)
            self.dr2 = nn.Dropout()
            self.lsbp2 = nn.Linear_SBP(cfg[5], cfg[6])
            self.rel3 = nn.ReLU(inplace=True)
            #fixme
            self.linear = nn.Linear(cfg[6], classes)
           
            
            self.classifier = [self.dr1, 
                               self.lsbp1, 
                               self.rel2,
                               self.dr2,
                               self.lsbp2,
                               self.rel3,
                               self.lsbp3]
        self.features = [self.block1, 
                         self.mp1, 
                         self.block2,
                         self.mp2,
                         self.block3,
                         self.block4,
                         self.block5,
                         self.mp3]
        
        if kl_weights:
            self.kl_weights = kl_weights
    def forward(self, x):

         
        if self.training:
            kl_temp = 0
            x, kl1 = self.block1(x)
             
            
            x = self.mp1(x)

            x, kl2 = self.block2(x)
            
            x = self.mp2(x)
                
            x, kl3 = self.block3(x)

            x, kl4 = self.block4(x)
            x, kl5 = self.block5(x)

            x = self.mp3(x)
            
        
            kl_temp = kl1 * self.kl_weights[0] + kl2 * self.kl_weights[1]+ kl3 * self.kl_weights[2]+ kl4 * self.kl_weights[3]+ kl5 * self.kl_weights[4]
                       
            x = torch.flatten(x, 1)
            if not self.sbp_linear: 
                x = self.classifier(x)
                
            else: 
                x = self.dr1(x) 
                x, kl6 = self.lsbp1(x)

                x = self.rel2(x)
                x = self.dr2(x)
                x,kl7 = self.lsbp2(x)
                x = self.rel3(x)
                x =  self.linear(x)
                
                kl_temp = kl_temp + kl6 * self.kl_weights[5] + kl7 * self.kl_weights[6]
            return x, kl_temp
        else:

            x = self.block1(x)
            
            x = self.mp1(x)

            x = self.block2(x)
                
            x = self.mp2(x)
                

            x = self.block3(x)

            x = self.block4(x)
            
            x = self.block5(x)

            x = self.mp3(x)
            
        
            x = torch.flatten(x, 1)
            if not self.sbp_linear: 
                x = self.classifier(x)
                
            else: 
                x = self.dr1(x) 
                x = self.lsbp1
                x = self.rel2(x)
                x = self.dr2(x)
                x = self.lsbp2(x)
                x = self.rel3(x)
                x = self.linear(x)
            return x

        
    #get the layerwise sparsity
    def get_sparsity(self):
        spars = []
        snrs = []
        if self.conv:
            spars.append(self.block1.conv1.layer_sparsity())
            snrs.append(self.block1.conv1.display_snr())
            spars.append(self.block2.conv1.layer_sparsity())
            snrs.append(self.block2.conv1.display_snr())
            spars.append(self.block3.conv1.layer_sparsity())
            snrs.append(self.block3.conv1.display_snr())
            spars.append(self.block4.conv1.layer_sparsity())
            snrs.append(self.block4.conv1.display_snr())
            spars.append(self.block5.conv1.layer_sparsity())
            snrs.append(self.block5.conv1.display_snr())
         
        
        else: 
            spars.append(self.block1.spb1.layer_sparsity())
            snrs.append(self.block1.spb1.display_snr())
            spars.append(self.block2.spb1.layer_sparsity())
            snrs.append(self.block2.spb1.display_snr())
            spars.append(self.block3.spb1.layer_sparsity())
            snrs.append(self.block3.spb1.display_snr())
            spars.append(self.block4.spb1.layer_sparsity())
            snrs.append(self.block4.spb1.display_snr())
            spars.append(self.block5.spb1.layer_sparsity())
            snrs.append(self.block5.spb1.display_snr())
            
        print(spars)
        print(snrs)
         