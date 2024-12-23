import numbers
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        cond_size = 0

        self.conv = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.down = nn.ModuleList([DownBlock(128, 256, cond_size), DownBlock(256, 512, cond_size)])
        self.mid = DownBlock(512, 512, cond_size)
        self.up = nn.ModuleList([UpBlock(512, 256), UpBlock(256, 128)])
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        outs = []
        for block in self.down:
            out, x = block(x)
            outs.append(out)
        x, _ = self.mid(x)
        for block in self.up:
            x = block(outs.pop(), x)

        x = self.conv_out(x)
        return x


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e # NOTE the size of codebook, similar to vocabulary size
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim) # NOTE codebook, can be learned
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. " f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # NOTE find the closest code in codebook for each z, e.g. z: (1998, 3), e: (256, 3)
        # NOTE min_encoding_indices: (1998) [0, 1, 23, 255, 255, ....]
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape) # NOTE replace with the value in codebook
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        # NOTE the quantized zq must as close as possible to the original z
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q: torch.Tensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.Tensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VQModel(nn.Module):
    def __init__(
        self,
        latent_channels: int = 1,
        num_vq_embeddings: int = 256,
        vq_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = UNet()

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = UNet()

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        h = self.encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h,)

        return h

    def decode(self, h: torch.Tensor, return_loss: bool = True, shape=None) -> torch.Tensor:
        # also go through quantization layer
        quant, commit_loss, _ = self.quantize(h)

        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2)

        if return_loss:
            return dec, commit_loss

        return dec

    def forward(self, sample: torch.Tensor, return_loss: bool = True) -> Tuple[torch.Tensor, ...]:
        r"""
        The [`VQModel`] forward method.

        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """

        h = self.encode(sample)  # latents
        dec, commit_loss = self.decode(h)

        if return_loss:
            return dec, commit_loss
        return dec
