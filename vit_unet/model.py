import torch.nn as nn
import torch 
from .VIT_model import *
# init_channel=3    编码器
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet_Encoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.encoder = nn.ModuleList()

        for out_channels in channels:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
            )
            in_channels = out_channels

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


# init_channel=512 解码器
class Unet_Decoder(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super().__init__()
        self.decoder = nn.ModuleList()
        for out_channels in decoder_channels:
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

    def forward(self, x):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

# class Unet(nn.Module):
#     def __init__(self, in_channels,channels):
#         super().__init__()
#         self.encoder=Unet_Encoder(in_channels,channels)
#     def forward(self,x):
#        return self.encoder(x)

class Vit_Unet(nn.Module):
    def __init__(self,in_channels,encoder_channels,decoder_channels,image_sizes,vit_dim,vit_depth,vit_heads,vit_mlp_dim,n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels=in_channels
        self.decoder_channels=decoder_channels
        self.inc=DoubleConv(in_channels,64)
        self.encoder = Unet_Encoder(64,encoder_channels[1:])
        self.decoder=Unet_Decoder(1024,decoder_channels) 
        self.vits = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i] * 2  # 假设跳跃连接需要将通道数翻倍
            out_ch = decoder_channels[i]
            self.convs.append(DoubleConv(in_ch, out_ch))
        for channel,image_size in zip(encoder_channels,image_sizes):   
            self.vits.append(ViT(image_size=image_size,patch_size=4,in_channels=channel,dim=vit_dim,depth=vit_depth,heads=vit_heads,mlp_dim=vit_mlp_dim))
        self.final_conv = nn.Conv2d(encoder_channels[0], 2, kernel_size=1)
       
    def forward(self,x):
        #编码器
        x=self.inc(x)
        encoder_features=[]
        encoder_features.append(x)
        
        encoder_features += self.encoder(x)
       
        
        vit_features = []
        #vit处理
        for feat, vit in zip(encoder_features, self.vits):
            vit_features.append(vit(feat))
        #解码器
        x=vit_features[-1]
        
        for i in range (len(self.decoder.decoder)):
            x = self.decoder.decoder[i](x)
           # self.conv = DoubleConv(x.shape[1]*2,self.decoder_channels[i])
            print(x.shape[1])
            if i < len(vit_features) -1:
                x = torch.cat([x, vit_features[-(i+2)]], dim=1)
                x=self.convs[i](x)
        # 最终输出
        x = self.final_conv(x)
        return x
    
    
    # def use_checkpointing(self):
    #     """启用梯度检查点以减少显存占用"""
    #     编码器部分
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     for i, layer in enumerate(self.encoder.encoder):
    #         self.encoder.encoder[i] = torch.utils.checkpoint(layer)
        
    #     ViT部分
    #     for i, vit in enumerate(self.vits):
    #         self.vits[i] = torch.utils.checkpoint(vit)
        
    #     解码器部分
    #     for i, layer in enumerate(self.decoder.decoder):
    #         self.decoder.decoder[i] = torch.utils.checkpoint(layer)
        
         

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    model=Vit_Unet(
        in_channels=3,
        encoder_channels=[64,128,256,512,1024],
        decoder_channels=[512,256,128,64],
        image_sizes=[256,128,64,32,16],
        vit_dim=1024,
        vit_depth=6,
        vit_heads=8,
        vit_mlp_dim=2048,
        n_classes=2
    ).to(device)
   
    x = torch.randn(1, 3, 256, 256).to(device) 
    #x = torch.randn(1, 3, 256, 256) 
    with torch.autograd.set_detect_anomaly(True):
    
        output=model(x)
    print(output.shape)

