import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from tqdm import tqdm

from diffusion.autoencoders import VQModel
from diffusion.loss import DiffusionHybridLoss
from diffusion.models import CFGuidance, DDPModule, UNet
from diffusion.predictor import NoisePredictor
from diffusion.schedule import DiscreteGaussianSchedule, linear_beta_schedule
from diffusion.transformers import RandomDiffusionSteps

WITH_VAE = True

def main():
    schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))
    predictor = NoisePredictor(schedule, lambda x: torch.clamp(x, -1, 1))

    # prepare model
    unet = UNet(time_size=32, digit_size=32)
    unet = CFGuidance(unet, 32, guidance=2.0)
    model = DDPModule(unet, schedule, predictor)
    encoder = nn.Embedding(10, 32)

    epochs = 10

    device = "cuda:0"
    encoder.to(device)
    model.to(device)

    if WITH_VAE:
        vqvae = VQModel().to(device)
        vqvae.load_state_dict(torch.load("vae.pt", map_location=device))

    # prepare data
    diffusion_transform = RandomDiffusionSteps(schedule, batched=True)
    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = FashionMNIST("fashion_mnist", train=True, download=True, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=2, pin_memory=True)

    # Apply optimizer to diffusion model and encoder for joint training
    optimizer = torch.optim.AdamW([{"params": encoder.parameters()}, {"params": model.parameters()}], lr=0.0001)
    # Define loss
    h_loss = DiffusionHybridLoss(schedule)

    encoder.train()
    model.train()
    for e in range(epochs):
        for sample in (pbar := tqdm(train_dataloader)):
            x, c = sample

            if WITH_VAE:
                latents = vqvae.encode(x)
            else:
                latents = x # the image itself

            noisy_latents = diffusion_transform({"x": latents}) # add noise
            x0, xt, noise, t, c = (
                noisy_latents["x"].to(device),
                noisy_latents["xt"].to(device),
                noisy_latents["noise"].to(device),
                noisy_latents["t"].to(device),
                c.to(device),
            )
            optimizer.zero_grad()

            # Compute loss
            embedding = encoder(c)
            out = model(xt, t, embedding)
            loss = h_loss(out.prediction, noise, out.mean, out.log_variance, x0, xt, t)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"{e+1}| Loss: {loss.item()}")

    # save model
    torch.save(model.state_dict(), "model.pt")
    torch.save(encoder.state_dict(), "encoder.pt")


if __name__ == "__main__":
    main()
