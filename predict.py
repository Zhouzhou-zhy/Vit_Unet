import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
#from unet import UNet
from vit_unet import Vit_Unet
from utils.utils import plot_img_and_mask


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        type=str,
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--output", "-o", metavar="OUTPUT", nargs="+", help="Filenames of output images"
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
    )
    parser.add_argument(
        "--no-save", "-n", action="store_true", help="Do not save the output masks"
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Maximum number of images to process and display",
    )
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    将掩码以指定颜色和透明度叠加到原图上
    Args:
    image (PIL.Image): 原始图像（RGB模式）
    mask (np.ndarray): 预测的掩码（0-1二值或多类索引）
    color (tuple/str): 掩码颜色，如 (255,0,0) 或 "red"
    alpha (float): 透明度 (0~1)
    Returns:
    PIL.Image: 叠加后的图像
    """
    # 将原图转换为RGBA模式以便添加透明度
    overlay = image.convert("RGBA")

    # 创建颜色层
    color_layer = Image.new("RGBA", overlay.size, color=color)

    # 根据掩码生成透明度层：掩码区域为alpha，其他区域为0
    if mask.ndim == 2:  # 二值掩码
        mask_array = (mask > 0).astype(np.uint8) * int(255 * alpha)
    else:  # 多分类掩码（假设mask为类别索引）
        mask_array = (mask > 0).astype(np.uint8) * int(255 * alpha)

    mask_image = Image.fromarray(mask_array, mode="L")

    # 将颜色层与原图混合
    overlay = Image.composite(color_layer, overlay, mask_image)

    # 合并图层并转换回RGB模式
    return overlay.convert("RGB")


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]
    in_files = in_files[: args.max_images]  # 限制最多处理 args.max_images 张图像
    out_files = get_output_filenames(args)

    #net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net=Vit_Unet(
        in_channels=3,
        encoder_channels=[64,128,256,512,1024],
        decoder_channels=[512,256,128,64],
        image_sizes=[256,128,64,32,16],
        vit_dim=1024,
        vit_depth=1,
        vit_heads=16,
        vit_mlp_dim=2048,
        n_classes=2,
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    # 创建一个大的 plt 画布
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # 4x4 网格，显示 16 张图像
    axes = axes.ravel()  # 将二维数组展平为一维，方便遍历

    for i, filename in enumerate(in_files):
        if i > args.max_images:  # 确保不超过最大图像数量
            break

        logging.info(f"Predicting image {filename} ...")
        img = Image.open(filename)

        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )

        result = overlay_mask(img, mask, color="red", alpha=1)

        axes[i].imshow(result)
        axes[i].set_title("test")  # 显示文件名作为标题
        axes[i].axis("off")  # 不显示坐标轴

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f"Mask saved to {out_filename}")

        if args.viz:
            logging.info(
                f"Visualizing results for image {filename}, close to continue..."
            )
            plot_img_and_mask(img, mask)
    # 调整子图间距并显示
    plt.tight_layout()
    plt.show()
