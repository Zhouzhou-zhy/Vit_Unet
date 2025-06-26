from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone.mobilevit_bone import MobileViTBackbone
from .backbone.mobilevit import mobile_vit_small, mobile_vit_x_small, mobile_vit_xx_small
from .backbone.mvit_unet_backbone import MobileVitUnetBackbone
from .backbone import (resnet,)
from .backbone.mobilevit_unet import Vit_Unet


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier,n_classes=num_classes,bilinear=False)
    return model



def _segm_mobilevit(name, backbone_name, num_classes, mobilevit_size,pretrained_backbone=False):
    if mobilevit_size == 'xx_small':
        backbone = MobileViTBackbone(mobile_vit_xx_small(num_classes=num_classes))
    elif mobilevit_size == 'x_small':
        backbone = MobileViTBackbone(mobile_vit_x_small(num_classes=num_classes))
    else:
        backbone = MobileViTBackbone(mobile_vit_small(num_classes=num_classes))
    inplanes =  640  # 取决于你最后的通道数
    low_level_planes = 64  # 取决于你 layer1 的输出通道数

    if name == 'deeplabv3plus':
        return_layers = {'conv_1x1_exp': 'out', 'layer2': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, [6, 12, 18])
    elif name == 'deeplabv3':
        return_layers = {'conv_1x1_exp': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, [6, 12, 18])
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier,n_classes=num_classes,bilinear=False)
    return model

def _segm_mvit_unet( num_classes):
    backbone=MobileVitUnetBackbone(Vit_Unet(n_channels=3, n_classes=num_classes, bilinear=False))
    inplanes =  512  # 取决于你最后的通道数
    low_level_planes = 64  # 取决于你 layer1 的输出通道数
    return_layers = {'mobilevit': 'out', 'layer7': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, [6, 12, 18])
    #backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier,n_classes=num_classes,bilinear=False)
    return model


def deeplabv3_resnet50(num_classes=8, output_stride=8, pretrained_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _segm_resnet('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilevit(num_classes=8,  pretrained_backbone=False):
    """Constructs a DeepLabV3+ model with a mobilevit backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _segm_mobilevit('deeplabv3plus', 'MobileViT_16', num_classes=8, mobilevit_size='',pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mvit_unet(num_classes=8):
     """Constructs a DeepLabV3+ model with a mobilevit_unet backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
     return _segm_mvit_unet(num_classes=num_classes)

if __name__ == '__main__':
    # 创建使用ResNet-101的DeepLabV3+模型
    model = deeplabv3plus_mvit_unet(num_classes=2)
    
    # 测试前向传播
    import torch
    input_tensor = torch.randn(2, 3, 256, 256)  # (batch_size, channels, height, width)
    output = model(input_tensor)
    
    #print(f"模型结构:\n{model}")
   # print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")  # DeepLabV3+ 输出是字典格式
