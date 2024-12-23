import torch
import torchvision
from torch import nn

from diffusion.autoencoders import VQModel
from diffusion.models import CFGuidance, DDPModule, UNet
from diffusion.predictor import NoisePredictor
from diffusion.schedule import DiscreteGaussianSchedule, linear_beta_schedule

WITH_VAE = True

def main():
    def fashion_encoder(name, num=1):
        # tokenizer
        fashion_dict = {"t-shirt": 0, "pants": 1, "sweater": 2, "dress": 3, "coat": 4,
                        "sandal": 5, "shirt": 6, "sneaker": 7, "purse": 8, "boot": 9}
        idx = torch.as_tensor([fashion_dict[name] for _ in range(num)]).to(device)

        encoder.eval()
        with torch.no_grad():
            embed = encoder(idx)
        return embed

    schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))
    predictor = NoisePredictor(schedule, lambda x: torch.clamp(x, -1, 1))
    device = "cuda:0"

    if WITH_VAE:
        vqvae = VQModel().to(device)
        vqvae.load_state_dict(torch.load("vae.pt", map_location=device))
        vqvae.eval()
        unet = UNet(input_channel=64, time_dim=32, digit_dim=32) # digit_dim is the text dim
    else:
        unet = UNet(input_channel=1, time_dim=32, digit_dim=32)

    unet = CFGuidance(unet, 32, guidance=2.0)
    model = DDPModule(unet, schedule, predictor).to(device)
    encoder = nn.Embedding(10, 32).to(device) # 10： 0-9

    model.load_state_dict(torch.load("model.pt", map_location=device))
    encoder.load_state_dict(torch.load("encoder.pt", map_location=device))

    model.eval()
    encoder.eval()

    c = fashion_encoder("boot", 9)
    noise = torch.randn(size=(9,1,32,32)).to(device)

    if WITH_VAE:
        noise_latents = vqvae.encode(noise) # to latent space
    else:
        noise_latents = noise # the latent is exactly the noise

    with torch.no_grad():
        img_latents = model(noise_latents, conditional_inputs=c)

        if WITH_VAE:
            imgs = vqvae.decode(img_latents, return_loss=False)
        else:
            imgs = img_latents # the image is exactly the diffuson output

    img_grid = torchvision.utils.make_grid(imgs, 3)
    img = torchvision.transforms.functional.to_pil_image((img_grid + 1) / 2)
    img.resize((288, 288))
    img.save("output.png")


if __name__ == "__main__":
    main()
