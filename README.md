# SSL Experiments

Self-supervised learning experiments with SimCLR, MoCo, and BYOL on image datasets.

## Setup

```bash
conda create -n ssl python=3.11 -y
conda activate ssl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install lightly wandb timm einops matplotlib scikit-learn tqdm ipykernel jupyter tensorboard
```

## Experiments

- [ ] Experiment 1: SimCLR on CIFAR-10 (warm-up)
- [ ] Experiment 2: SimCLR on STL-10 with ablation study
- [ ] Experiment 3: TBD (portfolio project)

## Hardware

- GPU: NVIDIA RTX 5070 Laptop (8GB VRAM, Blackwell sm_120)
- CUDA: 13.0 / PyTorch 2.13
- OS: WSL2 Ubuntu on Windows 11
