import torch.nn as nn
from .unet_parts import *
from .mobilevit import MobileViTBlock

class  Vit_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,vit_depth=6, vit_heads=8,base_cn=32):
        super(Vit_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, base_cn))
       
        self.down1 = (Down(base_cn, base_cn*2))
        self.down2 = (Down(base_cn*2, base_cn*4))
        self.down3 = (Down(base_cn*4, base_cn*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(base_cn*8,base_cn*16 // factor))
        self.mobilevit = MobileViTBlock(
            in_channels=base_cn*16,
            transformer_dim=base_cn*16,  # 与原设计保持比例
            ffn_dim=base_cn*16*4,  # 原4096=1024*4，保持比例系数
            n_transformer_blocks=vit_depth,
            patch_h=16,
            patch_w=16,
            head_dim=(base_cn*16) // vit_heads,  # 自动计算维度
            conv_ksize=3,
            dropout=0.1
        )
        
        self.up1 = (Up(base_cn*16, base_cn*8 // factor, bilinear))
        self.up2 = (Up(base_cn*8, base_cn*4 // factor, bilinear))
        self.up3 = (Up(base_cn*4, base_cn*2 // factor, bilinear))
        self.up4 = (Up(base_cn*2, base_cn, bilinear))
        self.outc = (OutConv(base_cn, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        vit_out = self.mobilevit(x5)    
        x = self.up1(vit_out, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256) 
    model=Vit_Unet(3,2)
    output=model(x)
    print(output.shape)
    
    