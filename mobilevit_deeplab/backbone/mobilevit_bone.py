#from  mobilevit import mobile_vit_small, mobile_vit_x_small, mobile_vit_xx_small
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as Fs
class MobileViTBackbone(nn.Module):
    def __init__(self, mobilevit_model):
        super().__init__()
        
        # 提取MobileViT的中间层特征
        self.stem = mobilevit_model.conv_1
        self.layer1 = mobilevit_model.layer_1
        self.layer2 = mobilevit_model.layer_2
        self.layer3 = mobilevit_model.layer_3
        self.layer4 = mobilevit_model.layer_4
        self.layer5 = mobilevit_model.layer_5
        
        # 通道调整
        self.channel_adjust = nn.Conv2d(512, 2048, 1)  # 根据实际输出调整

    def forward(self, x):
        # 前向传播获取多级特征
        x = self.stem(x)       # [B, 16, 112, 112]
        x = self.layer1(x)    # [B, 32, 56, 56]
        x = self.layer2(x)   # [B, 64, 28, 28]
        x = self.layer3(x)   # [B, 128, 14, 14]
        x = self.layer4(x)   # [B, 256, 7, 7]
        x = self.layer5(x)   # [B, 512, 7, 7]
        x=self.channel_adjust(x)   # [B, 2048, 7, 7]
        # 返回高层和低层特征
        return x

      
            
