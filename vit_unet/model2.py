import torch.nn as nn
from .unet_parts import *
#from .mobilevit import MobileViTBlock
from .VIT_model import ViT
class  Vit_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,vit_dim=512,vit_depth=6, vit_heads=8):
        super(Vit_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
       
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.vit = ViT(
            image_size=16,        # 假设输入特征图尺寸为16x16
            patch_size=4,        # 将16x16分成4x4的patch
            in_channels=1024,     # 与down4输出通道一致
            dim=vit_dim,          # ViT隐层维度
            depth=vit_depth,      # ViT层数
            heads=vit_heads,      # 注意力头数
            mlp_dim=vit_dim*2     # MLP扩展维度
        )
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        vit_out = self.vit(x5)    
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
    
    