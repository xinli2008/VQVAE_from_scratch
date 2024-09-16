import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from model import VQVAE
import os

def get_dataloader(dataset_type, batch_size, img_shape, is_train = True):
    if dataset_type == "MNIST":
        # 加载MNIST数据集
        custom_transform = transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        dataset = datasets.MNIST(root = "./data", train = is_train, transform = custom_transform, download = True)
    else:
        raise ValueError(f"unsupported type for {dataset_type}")

    # 创建dataloader
    # dataloader的num_workers参数决定了数据加载时使用的工作进程的数量
    # 也就是在训练过程中, 多少个并行的进程同时负责从磁盘来加载数据
    dataloader = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4)
    return dataloader


def train_vqvae(
    model: VQVAE, 
    img_shape = None,
    device = "cuda",
    batch_size = 64,
    dataset_type = "MNIST",
    save_model_pth = "/mnt/VQVAE_from_scratch/saved_model",
    lr = 1e-3,
    num_epochs = 100,
    loss_embedding_weight = 1,
    loss_commitment_weight = 0.25,
    log_dir = "logs/vqvae_experiment"
    ):

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir = log_dir)

    # 加载数据集
    dataloader = get_dataloader(dataset_type, batch_size = batch_size, img_shape = img_shape)

    # 模型加载到指定设备
    model.to(device)
    
    # 将模型设置为训练模式
    model.train()

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        # 初始化模型的loss
        total_loss = 0.0

        for i, (x, _) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)

            # 前向传播
            x_hat, ze, zq = model(x)

            # NOTE:损失计算
            # VQVAE的loss主要分为三个部分: 重建loss、嵌入loss和承诺loss
            # 重建loss: 原始图像x和重建图像x_hat之间的差异。
            # 嵌入loss：量化后的zq和ze之间的差异。
            # 承诺loss: 使ze接近zq
            loss_reconstruct = mse_loss(x, x_hat)
            loss_embedding = mse_loss(ze.detach(), zq)
            loss_commitment = mse_loss(ze, zq.detach())
            loss = loss_reconstruct + loss_embedding_weight * loss_embedding +  loss_commitment_weight * loss_commitment

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            total_loss += loss.item() * current_batch_size

            # 将每个batch的损失记录到tensorboard中
            writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataloader) + i)

        # 平均损失
        total_loss /= len(dataloader.dataset)
        toc = time.time()  # 记录每个epoch结束的时间
        
        # 保存模型
        if epoch % 10 ==0:
            model_path = f"vqvae_epoch{epoch}.pth"
            ckpt_path = os.path.join(save_model_pth, model_path)
            torch.save(model.state_dict(), ckpt_path)
        
        # 打印每个epoch的损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')

        # 将每个 epoch 的总损失记录到 TensorBoard
        writer.add_scalar('Loss/train_epoch', total_loss, epoch)
    
    writer.close()  # 关闭 TensorBoard 日志
    print('Training Complete')


if __name__ == "__main__":
    model = VQVAE(input_dim = 1, dim = 64, n_embedding = 512)
    train_vqvae(model, 
                img_shape = (28, 28), 
                device = "cuda", 
                batch_size = 64, 
                dataset_type = "MNIST", 
                save_model_pth = "/mnt/VQVAE_from_scratch/saved_model",
                lr = 1e-3,
                num_epochs = 100,
                )