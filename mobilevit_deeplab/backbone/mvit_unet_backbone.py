#from  mobilevit import mobile_vit_small, mobile_vit_x_small, mobile_vit_xx_small
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as Fs
class MobileVitUnetBackbone(nn.Module):
    def __init__(self, mobilevit_model):
        super().__init__()
        
        # 提取MobileViT的中间层特征
        self.stem = mobilevit_model.inc
        self.layer1 = mobilevit_model.down1
        self.layer2 = mobilevit_model.down2
        self.layer3 = mobilevit_model.down3
        self.layer4 = mobilevit_model.down4
        self.mobilevit=mobilevit_model.mobilevit
        self.layer5=mobilevit_model.up1
        self.layer6=mobilevit_model.up2
        self.layer7=mobilevit_model.up3
        self.layer8=mobilevit_model.up4
        self.layer9=mobilevit_model.outc

    def forward(self, x):
        # 前向传播获取多级特征
        x = self.stem(x)      
        x = self.layer1(x)   
        x = self.layer2(x)   
        x = self.layer3(x)   
        x = self.layer4(x)
        x=self.mobilevit(x)
        x = self.layer5(x)  
        x = self.layer6(x)   
        x = self.layer7(x)   
        x = self.layer8(x)   
        x=self.layer9(x)
        # 返回高层和低层特征
        return x

      
            
