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
        custom_transform = transforms.Compose([
            transforms.Resize(img_shape),       # 默认使用的是双线性插值
            transforms.ToTensor(),              # 将numpy或者PIL转化为tensor, 并且将图像像素从[0,255]转化为[0,1]之间
            transforms.Normalize((0.5), (0.5))  # transformers.normalize传入的是值是均值和标准差, 因为均值和标准差只传入了一个0.5,这意味着假定图像只有一个颜色通道(如灰度图),或者对所有通道使用相同的值。
            # NOTE: 标准化的计算公式为：(input[channel] - mean[channel]) / std[channel]。在这个例子中，每个通道的像素值将从[0.0, 1.0]转换到[-1.0, 1.0]。
        ])
        dataset = datasets.MNIST(root = "./data", train = is_train, transform = custom_transform, download = True)
    else:
        raise ValueError(f"unsupported type for {dataset_type}")

    r"""
    创建Dataloader对象
    参数:
        dataset: 数据集对象
        batch_size: 每个batch的大小
        shuffle: 是否在每个epoch开始时打乱数据
        num_workers: 用于数据加载的子进程数量(0表示主进程加载)
        collate_fn: 用于将样本列表合并为一个batch的函数
        sampler: 自定义采样器, 与shuffle互斥
        batch_sampler: 自定义batch采样器, 与batch_size, shuffle, sampler互斥
        pin_memory: 是否将数据加载到GPU的固定内存中, 以加快数据传输速度
        drop_last: 如果数据集大小不能被batch_size整除, 是否丢弃最后一个不完整的batch
        timeout: 数据加载的超时时间(秒)
    returns:
        dataloader对象  
    """
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

    dataloader = get_dataloader(dataset_type, batch_size = batch_size, img_shape = img_shape)
    model.to(device)
    model.train()

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (x, _) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)

            # NOTE：VQVAE的前向传播会返回三个值: 重构的x_hat, 编码器输出ze, 量化后的zq
            x_hat, ze, zq = model(x)

            # NOTE: VQVAE的损失函数由三部分组成: 重构损失, 嵌入损失, 和承诺损失
            # 1. 重构损失: 衡量的是输入x和重构x_hat之间的差异, 通常使用均方误差(MSE)来计算
            # 2. 嵌入损失： 衡量的是编码器输出ze和量化后的zq之间的差异, 通过最小化这个损失, 可以确保编码器输出接近于量化后的表示
            # 3. 承诺损失: 衡量的是编码器输出ze和量化后的zq之间的差异, 通过最小化这个损失, 可以确保量化后的表示不会偏离编码器输出太远

            # NOTE：可以看到后面两个loss虽然计算的是相同的MSE值，但是梯度流向完全不同。
            loss_reconstruct = mse_loss(x, x_hat)
            # 1. loss_embedding = mse_loss(ze.detach(), zq)：梯度只流向了zq，因为ze被detach了，所以它是更新码本，使得码本向量zq靠近编码器输出ze。
            # 2.  loss_commitment = mse_loss(ze, zq.detach())：梯度只流向了ze，因为zq被detach了，所以它是更新编码器，使得编码器输出ze向量向码本向量zq靠近。
            # 这两个损失项共同作用，使得编码器和码本相互靠近，从而实现有效的向量量化。
            loss_embedding = mse_loss(ze.detach(), zq)
            loss_commitment = mse_loss(ze, zq.detach())
            # NOTE: 如果只有1个方向的loss的话，比如说只有loss_embedding的话, 则编码器可以随意输出，码本疲于追赶，训练不稳定。
            # 同理，如果只有loss_commitment的话，码本可以随意输出，编码器疲于追赶，也会导致训练不稳定。

            loss = loss_reconstruct + loss_embedding_weight * loss_embedding +  loss_commitment_weight * loss_commitment

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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