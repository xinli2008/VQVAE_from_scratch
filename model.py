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
        self.encoder = nn.Sequential(
                                nn.Conv2d(input_dim, dim, 4, 2, 1),
                                nn.ReLU(),
                                # NOTE: 关于nn.ReLU中inplace参数的解释：
                                # inplace = False(默认): 不在原地操作, 会创建并返回一个姓的张量, 输入张量保持不变。
                                # inplace = True, 在原地操作, ReLU直接修改输入张量, 不会创建新的张量。优点是节省内存。
                                nn.Conv2d(dim, dim, 4, 2, 1),
                                nn.ReLU(), 
                                nn.Conv2d(dim, dim, 3, 1, 1),
                                ResidualBlock(dim), 
                                ResidualBlock(dim)
                                )
        
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
                                    ResidualBlock(dim), 
                                    ResidualBlock(dim),
                                    nn.ConvTranspose2d(dim, dim, 4, 2, 1), 
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)
                                    )
        self.n_downsample = 2

    def forward(self, x):
        # 编码器部分
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape

        # 通过广播机制，我们可以将每个编码特征ze与所有的嵌入向量进行比较
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        
        # NOTE: 计算VQ-VAE中每个编码特征向量ze和所有离散的码本codebook嵌入向量之间的距离, 并找到距离最近的码本向量, 也就是量化过程中选择最近邻的过程。
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        # NOTE: nearest_neighbor返回的是一个[N,H,W]的向量, 它包含了每个编码特征在码本中最接近的嵌入向量的索引。
        # 具体来说, argmin会沿着第1个维度(即K维度,表示码本中嵌入向量的数量)寻找最小的值。
        nearest_neighbor = torch.argmin(distance, 1)
        
        # NOTE: 通过 nearest_neighbor 索引从 vq_embedding 中查找最近的离散嵌入向量，得到一个形状为 [N, H, W, C] 的张量 zq
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        
        # NOTE: 在VQ-VAE中, 输入线经过encoder得到ze, 在经过codebook得到量化后的zq, zq在经过decoder得到重建后的信息。
        # 但是, zq是不可微的, 所以无法直接通过zq计算反向传播的梯度。
        # NOTE: detach()是pytorch中用于从计算图中分离张量的方法, 表示这个张量在计算反向传播时不会参与梯度计算。
        # 也就是说, 通过detach, 我们可以控制zq-ze这部分不会参数梯度计算。
        # 所以说, 使用了下面这部分代码, decoder_input首先接受到了正确的信息zq, 并且编码器可以继续接收梯度更新并进行训练。
        decoder_input = ze + (zq - ze).detach()

        # 解码器部分
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