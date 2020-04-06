import torch
import torch.nn as nn

from SBP_utils_gpu import Conv2d_SBP, Linear_SBP, SBP_layer 

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


class SBPConv_AlexNet(nn.Module):

    def __init__(self, cfg, classes=100,kl_weights= generic_kl_weights['eight']):
        super(SBPConv_AlexNet, self).__init__()
        
        self.cfg = cfg
        
        self.kl_weights = kl_weights
        
        self.block1 = SBP_ConvBlock(3,cfg[0])
        self.mp1 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.block2 = SBP_ConvBlock(cfg[0],cfg[1],kernel_size=3,stride=1,padding=1)
        self.mp2= nn.MaxPool2d(kernel_size=3,stride=2)

        self.block3 = SBP_ConvBlock(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1)
        self.block4 = SBP_ConvBlock(cfg[2],cfg[3],kernel_size=3, stride=1, padding=1)
        self.block5 = SBP_ConvBlock(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1)

            
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
       
      
       

        self.dr1 = nn.Dropout()
        self.lsbp1 =  Linear_SBP(cfg[4] * 1 * 1, cfg[5])
        self.rel2 = nn.ReLU(inplace=True)
        self.dr2 = nn.Dropout()
        self.lsbp2 = Linear_SBP(cfg[5], cfg[6])
        self.rel3 = nn.ReLU(inplace=True)
       
        self.last = nn.Linear(cfg[6], classes)


        self.classifier = [self.dr1, 
                           self.lsbp1, 
                           self.rel2,
                           self.dr2,
                           self.lsbp2,
                           self.rel3,
                           self.last]
        
        self.features = [self.block1, 
                         self.mp1, 
                         self.block2,
                         self.mp2,
                         self.block3,
                         self.block4,
                         self.block5,
                         self.mp3]
        

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
           
            #classifer
            x = self.dr1(x) 
            x, kl6 = self.lsbp1(x)
            x = self.rel2(x)
            x = self.dr2(x)
            x,kl7 = self.lsbp2(x)
            x = self.rel3(x)
            x =  self.last(x)
                
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
          
            x = self.dr1(x) 
            x = self.lsbp1(x)
            x = self.rel2(x)
            x = self.dr2(x)
            x = self.lsbp2(x)
            x = self.rel3(x)
            x = self.last(x)
            return x

        
    #get the layerwise sparsity
    def get_sparsity(self):
        spars = []
        snrs = []
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
        
        spars.append(self.lsbp1.layer_sparsity())
        spars.append(self.lsbp2.layer_sparsity())
        snrs.append(self.lsbp1.display_snr())
        snrs.append(self.lsbp2.display_snr())
         
         
        print("Sparsity: ",spars)
        print("SNRS: ", snrs)
         
    
    


         