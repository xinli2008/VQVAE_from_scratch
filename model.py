import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp

class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        
        # 创建了一个嵌入矩阵, 它的形状为[n_embedding, dim]。
        # 这个嵌入矩阵可以看做是一个查找表, 表中的每一行是一个长度为dim的嵌入向量, 共有n_embedding行。
        # NOTE: 在VQ-VAE中的作用:
        # 在前向传播过程中, 编码器会生成连续的特征表示ze, 然后根据这些特征从n_embedding个向量中找到最接近的一个,
        # 然后将其用作量化后的特征zq。
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2

    def forward(self, x):
        # encode
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape

        # 通过广播机制，我们可以将每个编码特征ze与所有的嵌入向量进行比较
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        
        # 每个编码特征ze与所有嵌入向量之间的欧式距离（即平方差的和）
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        
        # 我们知道在VQ-VAE的前向传播中, ze通过codebook得到zq, 此时它已经从连续的向量变化离散的向量。
        # 离散的量化操作不可微，因此无法直接通过zq计算反向传播的梯度
        # 通过使用 detach() 技巧, VQ-VAE能够在不依赖zq梯度的情况下让编码器继续接收梯度更新, 从而解决训练问题
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)

if __name__ == "__main__":
    input_dim = 3
    dim = 64
    n_embedding = 512

    vqvae = VQVAE(input_dim, dim, n_embedding)
    input_tensor = torch.randn(8, 3, 64, 64)
    output, ze, zq = vqvae(input_tensor)

    # 打印结果的形状
    print("输入张量的形状: ", input_tensor.shape)
    print("重建图像的形状: ", output.shape)
    print("编码特征 ze 的形状: ", ze.shape)
    print("量化特征 zq 的形状: ", zq.shape)