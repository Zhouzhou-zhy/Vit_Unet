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
from unet import UNet
from vit_unet import Vit_Unet
from utils.utils import plot_img_and_mask
from mobilevit_deeplab.modeling import deeplabv3plus_mvit_unet,deeplabv3_resnet50

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
        "--classes", "-c", type=int, default=3, help="Number of classes"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=48,
        help="Maximum number of images to process and display",
    )
    return parser.parse_args()
colors = [
        (255, 255, 255),        # 类别0 - 白色
        (255, 0, 0),      # 类别1 - 红色
        (0, 255,0),      # 类别2 - 绿色
        (255, 255, 0),      # 类别3 - 蓝色
        (0, 0, 255),    # 类别4 - 黄色
        (255, 69, 0),    # 类别5 - 紫红
        (128, 0, 128),    # 类别6 - 红色
    ]

def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_color_image(mask: np.ndarray):
    # 自定义每个类别的颜色（0~6）
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        color_mask[mask == i] = color

    return Image.fromarray(color_mask)



def overlay_multiclass_mask(image, mask, colors, alpha=0.5):
    """将多分类掩码叠加到图像上"""
    image = image.convert("RGBA")
    overlay = image.copy()

    for class_idx, color in enumerate(colors):
        if class_idx == 0:
            continue  # 跳过背景
        binary_mask = (mask == class_idx).astype(np.uint8) * int(255 * alpha)
        mask_image = Image.fromarray(binary_mask, mode="L")
        color_layer = Image.new("RGBA", image.size, color=color)
        overlay = Image.composite(color_layer, overlay, mask_image)

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
    # net=Vit_Unet(
    #     n_channels=3,
    #     n_classes=3,
        
    # )
    #net = UNet(n_channels=3, n_classes=256, bilinear=False)
    #net=deeplabv3plus_mvit_unet(num_classes=args.classes)
    net=deeplabv3plus_mvit_unet(num_classes=args.classes)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", list(range(args.classes)))
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    # 创建一个大的 plt 画布 
    batch_size = 16
    for batch_idx in range(0, len(in_files), batch_size):
        batch_files = in_files[batch_idx : batch_idx + batch_size]
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        for i, filename in enumerate(batch_files):
            # if i > args.max_images:  # 确保不超过最大图像数量
            #     break
                
            logging.info(f"Predicting image {filename} ...")
            img = Image.open(filename)

            mask = predict_img(
                net=net,
                full_img=img,
                scale_factor=args.scale,
                out_threshold=args.mask_threshold,
                device=device,
            )

            result = overlay_multiclass_mask(img, mask, colors=colors, alpha=0.5)

            axes[i].imshow(result)
            axes[i].set_title("test")  # 显示文件名作为标题
            axes[i].axis("off")  # 不显示坐标轴

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_color_image(mask)
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
