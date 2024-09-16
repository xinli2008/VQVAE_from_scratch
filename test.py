import torch
import os
import random
import numpy as np
import cv2
from main import get_dataloader
from model import VQVAE

def save_reconstruction(model, dataloader, device, num_images=10, output_path="./assets/vqvae_reconstruction.jpg"):
    # 模型切换到评估模式
    model.eval()
    model.to("cuda")

    # 创建输出文件夹
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 随机从dataloader中抽取num_images张图片
    all_images = []
    for x, _ in dataloader:
        all_images.append(x)
    all_images = torch.cat(all_images, dim=0)

    # 随机选取 num_images 张图片
    idx = random.sample(range(all_images.size(0)), num_images)
    test_images = all_images[idx].to(device)

    # 模型的前向传播
    with torch.no_grad():
        x_hat, _, _ = model(test_images)

    # 将原始图像和重建后的图像在高度方向上拼接
    comparison = torch.cat([test_images, x_hat], dim=2)  # dim=2 为垂直拼接

    # 转换为 numpy 数组
    comparison = comparison.cpu().permute(0, 2, 3, 1).numpy()  # 形状变为 [N, H, W, C]
    comparison = (comparison * 255).clip(0, 255).astype(np.uint8)

    # 水平拼接所有图像对
    comparison_image = np.concatenate(comparison, axis=1)  # 在宽度方向上拼接

    # 保存图片
    cv2.imwrite(output_path, comparison_image)

    print(f'Reconstructed image saved at {output_path}')

if __name__ == '__main__':
    model = VQVAE(input_dim=1, dim=64, n_embedding=512)
    ckpt_path = "/mnt/VQVAE_from_scratch/saved_model/vqvae_epoch50.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

    # 加载测试集数据
    test_dataloader = get_dataloader(dataset_type='MNIST', batch_size=64, img_shape=(28, 28), is_train=False)

    # 测试重建能力，并保存图片
    save_reconstruction(model, test_dataloader, device='cuda', num_images=10, output_path='work_dirs/vqvae_reconstruction.jpg')
