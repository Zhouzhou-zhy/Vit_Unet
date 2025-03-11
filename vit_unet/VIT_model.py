import torch
import torch.nn as nn
from vit_pytorch import ViT
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # dim=1024, hidden_dim=2048
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) #inner_dim=h*d 

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 首先生成q,k,v
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
def pair(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)

    


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        in_channels,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.patch_size=patch_size
        self.image_size=image_size    
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim)
        )  # (1,65,1024)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_feature_map = nn.Linear(dim, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)  # 新增上采样
     
       

    def forward(self, img):  # img: (1, 3, 256, 256)
        x = self.to_patch_embedding(img)  # (1, 64, 1024)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)  # (1, 1, 1024)
        x = torch.cat((cls_tokens, x), dim=1)  # (1, 65, 1024)
        x += self.pos_embedding[:, : (n + 1)]  # (1, 65, 1024)
        x = self.dropout(x)  # (1, 65, 1024)
        x=self.transformer(x)  # (1, 65, 1024)
        x = self.to_feature_map(x)  # [B, num_patches, C]
        x = x[:, 1:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.image_size//self.patch_size, w=self.image_size//self.patch_size)
        x = self.upsample(x)
        return x


# if __name__ == "__main__":
 
#     v = ViT(  # 图像大小
#         image_size=256,
#         patch_size=32, 
#         in_channels=3,# patch大小（分块的大小） # imagenet数据集1000分类
#         dim=1024,  # position embedding的维度
#         depth=1,  # encoder和decoder中block层数是6
#         heads=16,  # multi-head中head的数量为16
#         mlp_dim=2048,
#         dropout=0.1,  #
#         emb_dropout=0.1,
#     )

#     img = torch.randn(1, 3, 256, 256)

#     preds = v(img)  # (1, 1000)
#     print(list(preds.shape))
