import torch
import torchvision
from torch import nn

from diffusion.models import CFGuidance, DDPModule, UNet
from diffusion.predictor import NoisePredictor
from diffusion.schedule import DiscreteGaussianSchedule, linear_beta_schedule


def main():
    def fashion_encoder(name, num=1):
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

    unet = UNet(time_size=32, digit_size=32)
    unet = CFGuidance(unet, 32, guidance=2.0)
    model = DDPModule(unet, schedule, predictor).to(device)
    encoder = nn.Embedding(10, 32).to(device)

    model.load_state_dict(torch.load("model.pt", map_location=device))
    encoder.load_state_dict(torch.load("encoder.pt", map_location=device))

    model.eval()
    encoder.eval()

    c = fashion_encoder("boot", 9)
    noise = torch.randn(size=(9,1,32,32)).to(device)

    with torch.no_grad():
        imgs = model(noise, conditional_inputs=c)

    img_grid = torchvision.utils.make_grid(imgs, 3)
    img = torchvision.transforms.functional.to_pil_image((img_grid + 1) / 2)
    img.resize((288, 288))
    img.save("output.png")


if __name__ == "__main__":
    main()
