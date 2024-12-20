# prepare model, optimizer

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from tqdm import tqdm

from diffusion.autoencoders import VQModel

vqvae = VQModel()
optimizer = torch.optim.AdamW(
    [{"params": vqvae.parameters()}], lr=0.0001
)

transforms = Compose([Resize((32, 32)), ToTensor()])
train_dataset = FashionMNIST("fashion_mnist", train=True, download=True, transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=2, pin_memory=True)


# train loop
epochs = 25
device = "cuda:0"
vqvae.to(device)

for e in range(epochs):
    for sample in (pbar := tqdm(train_dataloader)):
        x, c = sample
        x = x.to(device)
        x_out, commit_loss = vqvae(x, return_loss=True)
        loss = commit_loss + F.mse_loss(x_out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'{e+1}| Loss: {loss.item()}')

torch.save(vqvae.state_dict(), "vae.pt")
