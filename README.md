# VQVAE_from_scratch

## Introduction

Pytorch版本实现的基于MNIST的VQVAE(仅供学习)

VQ-VAE是一种改进的VAE,专门用于学习离散的潜在表示空间。VQ-VAE通过**引入向量量化(vector quantization)技术**，将连续的潜在向量映射到离散的嵌入空间。向量量化指的是输入在经过编码器后, 会被映射到一个离散的码本(codebook), 码本由一组离散的嵌入向量组成。VQ-VAE通过将每个连续特征向量与码本中的向量进行比较,选择最接近的嵌入向量(最近邻),这就是量化过程。

## Preliminary

- **VQVAE-architecture**

![vqvae](./assets/vqvae-architecture.png)

## Loss

![loss](./assets/vqvae_loss.png)

## Inference

![推理](./work_dirs/vqvae_reconstruction.jpg)

## Acknowledgements

- [轻松理解 VQ-VAE：首个提出 codebook 机制的生成模型](https://zhouyifan.net/2023/06/06/20230527-VQVAE/)
- [VQVAE预训练模型的论文原理及PyTorch代码逐行讲解](https://www.bilibili.com/video/BV14Y4y1X7wb/?spm_id_from=333.337.search-card.all.click)
- [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)
