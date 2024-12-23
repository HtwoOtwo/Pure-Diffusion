import copy
from dataclasses import dataclass
from typing import Callable, Dict, Generator, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .predictor import Predictor
from .schedule import DiscreteGaussianSchedule


@dataclass
class DiffusionOutput:
    prediction: Tensor
    variance_value: Optional[Tensor] = None
    mean: Optional[Tensor] = None
    log_variance: Optional[Tensor] = None


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + cond_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, c=None):
        _, _, w, h = x.size()
        if c is not None:
            c = c.expand(-1, -1, w, h)  # Shape conditional input to match image
            x = self.block(torch.cat([x, c], 1))  # Convolutions over image + condition
        else:
            x = self.block(x)
        x_small = self.pooling(x)  # Downsample output for next block
        return x, x_small


# Upscaling blocks on unet
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_res=True):
        super().__init__()
        if use_res:
            self.block = nn.Sequential(
                nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        self.upsample = nn.Upsample(scale_factor=2)
        self.use_res = use_res

    def forward(self, x_small, x=None):
        if self.use_res:
            assert x is not None
            x_big = self.upsample(x_small)  # Upscale input back towards original size
            x = torch.cat((x_big, x), dim=1)  # Join previous block with accross block
        else:
            x = self.upsample(x_small)
        x = self.block(x)  # Convolutions over image
        return x


class UNet(nn.Module):
    def __init__(self, input_channel=1, time_dim=32, digit_dim=32, steps=1000):
        super().__init__()
        cond_dim = time_dim + digit_dim

        self.conv = nn.Conv2d(input_channel, 128, kernel_size=3, padding=1)
        self.time_proj = nn.Embedding(steps, time_dim)
        self.var = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)
        self.down = nn.ModuleList([DownBlock(128, 256, cond_dim), DownBlock(256, 512, cond_dim)])
        self.mid = DownBlock(512, 512, cond_dim)
        self.up = nn.ModuleList([UpBlock(512, 256), UpBlock(256, 128)])

    def forward(self, x, t, conditional_inputs):
        b, _, _, _ = x.shape
        time_emb = self.time_proj(t).view(b, -1, 1, 1)
        cond = conditional_inputs.view(b, -1, 1, 1)
        cond = torch.cat((time_emb, cond), dim=1)

        x = self.conv(x)

        outs = []
        for block in self.down:
            out, x = block(x, cond)
            outs.append(out)
        x, _ = self.mid(x, cond)
        for block in self.up:
            x = block(x, outs.pop())

        v = self.var(x)
        p = self.pred(x)
        return DiffusionOutput(p, v)


class DDPModule(nn.Module):
    """DDPModule acts as a wrapper module around an inner neural network. During training it uses the
    inner neural network to predict a single denoising step. When set to eval, calling forward will
    sample the entire diffusion schedule. This module follows the denoising diffusion process as
    described in "Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2006.11239) and "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672).

    Example:
        ddpm_model = DDPModule(model, schedule, predictor)
        prediction_t = ddpm_model(x_t, t)

        ddpm_model.eval()
        x_0 = ddpm_model(x_T, T)

    Code ref:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

    Attributes:
        model (nn.Module): prediction neural network
        schedule (DiscreteGaussianSchedule): defines diffusion of noise through time
        predictor (Predictor): predictor class to handle predictions depending on the model input
        eval_steps (Tensor): subset of steps to sample at inference
        progress_bar (bool): whether to show a progress bar

    Args:
        x (Tensor): corrupted data at time t (when t = schedule.steps, x is equivalent to noise)
        timestep (Tensor): diffusion step
        conditional_inputs (Dict): dictionary of context embeddings

    """

    def __init__(
        self,
        model: nn.Module,
        schedule: DiscreteGaussianSchedule,
        predictor: Predictor,
        eval_steps: Optional[Tensor] = None,
        progress_bar: bool = True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.model = model
        self.schedule = schedule
        self.predictor = predictor
        self.progress_bar = progress_bar

        if eval_steps is None:
            eval_steps = torch.arange(self.schedule.steps)
            eval_steps_map = eval_steps
            self.eval_schedule = schedule
            self.eval_predictor = predictor
        else:
            # Special schedule for strided sampling from equation 19
            # in "Improved Denoising Diffusion Probabilistic Models"
            eval_steps, _ = eval_steps.sort()
            # eval_map maps from timestep in full schedule, to timestep in truncated eval scheule
            # e.g. if train has 1000 steps, and eval has 3, then t = 500 would map to eval timestep = 1
            eval_steps_map = torch.zeros(self.schedule.steps, dtype=torch.long)
            eval_steps_map[eval_steps] = torch.arange(len(eval_steps))

            # Compute cumulative product of only the alphas in the eval steps
            # Recompute betas based on these cumulative products and create a new schedule with these betas
            alphas_cumprod = schedule.alphas_cumprod[eval_steps]
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            beta_schedule = 1 - alphas_cumprod / alphas_cumprod_prev
            self.eval_schedule = copy.deepcopy(schedule)
            self.eval_schedule.betas = beta_schedule
            self.eval_predictor = copy.deepcopy(predictor)
            self.eval_predictor.schedule.betas = beta_schedule

        self.eval_steps: Tensor
        self.eval_steps_map: Tensor
        self.register_buffer("eval_steps", eval_steps.to(torch.long))
        self.register_buffer("eval_steps_map", eval_steps_map)

    def predict_parameters(self, input: DiffusionOutput, xt: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Given model predictions, corrupted data (x at step t) and noise (x at final step T),
        compute the predicted normal mean and log_variance for the given diffusion step t.

        Args:
            input (DiffusionOutput): model output values
            xt (Tensor): corrupted data at time t
            t (Tensor): int diffusion steps
        """
        pred, value = input.prediction, input.variance_value
        schedule = self.schedule if self.training else self.eval_schedule
        predictor = self.predictor if self.training else self.eval_predictor
        timestep = t if self.training else self.eval_steps_map[t]

        x0 = predictor.predict_x0(pred, xt, timestep)
        return schedule.q_posterior(x0, xt, timestep, value)

    def remove_noise(self, xt: Tensor, t: Tensor, c: Optional[Dict[str, Tensor]]) -> Tensor:
        """Given corrupted data (x at step t) and noise (x at final step T), compute x denoised
        by one diffusion step. This is the function p(xt) from
        https://arxiv.org/abs/2006.11239.

        Args:
            xt (Tensor): corrupted data at time t
            t (Tensor): int diffusion steps
            c (Dict): dictionary of context embeddings
        """
        # Model outputs
        out = self.model(xt, t, c)
        mean, log_variance = self.predict_parameters(out, xt, t)

        # Predict x_{t-1}
        dtype = xt.dtype
        noise = self.schedule.sample_noise(xt)
        # Mask noise when t = 0; shape (b, 1, ..., 1) with same dims as xt
        nonzero_mask = (t != 0).to(dtype).view(-1, *([1] * (xt.dim() - 1)))
        # pyre-ignore
        return mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def generator(self, x: Tensor, c: Optional[Dict[str, Tensor]] = None) -> Generator[Tensor, None, None]:
        """Generate xt for each t in sample_steps"""
        for step in self.eval_steps.flip(0):
            t = step * torch.ones(x.size(0), device=x.device, dtype=torch.long)
            x = self.remove_noise(x, t, c)
            yield x

    def forward(
        self,
        x: Tensor,
        timestep: Optional[Tensor] = None,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> Union[DiffusionOutput, Tensor]:
        if self.training:
            if timestep is None:
                raise ValueError("Must provide a t value during training")
            out = self.model(x, timestep, conditional_inputs)  # noise prediction
            if not isinstance(out, DiffusionOutput):
                raise TypeError("Model is expected to output a DiffusionOutput class")
            if out.variance_value is not None:
                out.mean, out.log_variance = self.predict_parameters(out, x, timestep)
            return out
        else:
            gen: Iterable = self.generator(x, conditional_inputs)  # inference
            if self.progress_bar:
                # Lazy import so that we don't depend on tqdm.
                from tqdm.auto import tqdm

                gen = tqdm(gen, total=len(self.eval_steps))
            for xi in gen:
                x = xi
            # pyre-ignore
            return x


class CFGuidance(nn.Module):
    """
    Classifier free guidance gives diffusion models the ability to sample from a conditional
    distribution, while maintaining a healthy ratio between exploitation (i.e. correlation
    with conditional variables) and exploration (i.e. diversity of generation).
    As described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided
    Diffusion Models" (https://arxiv.org/abs/2112.10741)

    During training, `p` controls how often does the model see/use the
    unconditional embeddings (i.e. zero or random or user-provided embedding).
    During inference, `guidance` guides what ratio of unconditional vs
    conditional embeddings guide the generative output. Additionally
    `eval_unconditional_embeddings` provides an option to specify an alternative
    embedding, other than the one used to train the model.

    Attributes:
        model (nn.Module): the neural network
        dim_cond (int): the dimensions of conditional embeddings
        p (float): probability values control
        guidance (float): a magnitude to control the strength of the context_embedding
            during inference. Higher values give better alignment with the context at the
            cost of lesser diversity. Expecting a value from 0 to inf, defaults to 0.
        learn_null_emb (bool): If False, then unconditional embeddings are set to zero and are not trainable
            If True, then unconditional embeddings are set to random and are trainable. Defaults to True.
        train_unconditional_embeddings (Optional[Tensor]): initial values to be used for
            unconditional embeddings for training. If not provided, random values or zero values are
            initialized. Defaults to None.
        eval_unconditional_embeddings (Optional[Tensor]): unconditional embeddings to be used for
            evaluation. If not provided, the learned unconditional embeddings will be used. Defaults to None.
    Args:
        x (Tensor): input Tensor of shape [b, in_channels, ...]
        timestep (Tensor): diffusion step
        conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
            Conditional embeddings must have at least 2 dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        dim_cond: int,
        p: Union[int, float, Dict[str, float]] = 0.1,
        guidance: float = 0.0,
        learn_null_emb: bool = True,
        train_unconditional_embeddings: Optional[Tensor] = None,
        eval_unconditional_embeddings: Optional[Tensor] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.model = model
        self.dim_cond = dim_cond
        self.p = p
        self.guidance = guidance
        self.learn_null_emb = learn_null_emb

        init_fn: Callable
        if self.learn_null_emb:
            init_fn = torch.rand
            requires_grad = True
        else:
            init_fn = torch.zeros
            requires_grad = False

        # Use user provided initial embeddings if provided, otherwise initialize with
        # zeros or random values
        self.train_unconditional_embedding = self._gen_unconditional_embeddings(train_unconditional_embeddings, init_fn, requires_grad)

        # Initialize eval embeddings with train embeddings
        # ParameterDict.copy() creates a new ParameterDict but keeps the references to
        # the same parameters as train. So any parameter updates should be avaialble here.
        if eval_unconditional_embeddings is None:
            self.eval_unconditional_embedding = self.train_unconditional_embedding.clone()

    def _gen_unconditional_embeddings(
        self,
        initial_embeddings: Optional[Dict[str, Tensor]],
        default_init_fn: Callable,
        requires_grad: bool,
    ) -> nn.ParameterDict:
        """
        Generate unconditional embeddings based on the dim values.
        Args:
            initial_embeddings (Tensor): initial embedding values,
                If embedding is not provided, then use `default_init_fn` to initialize.
            default_init_fn (Callable): function to initialize an embedding
            requires_grad (bool): whether or not to optimize this embedding
        Returns:
            unconditional_embeddings (Tensor): unconditional embeddings
        """
        if initial_embeddings:
            unconditional_embeddings = nn.Parameter(initial_embeddings, requires_grad=requires_grad)
        else:
            shape = (1,) + ((self.dim_cond,))
            unconditional_embeddings = nn.Parameter(default_init_fn(*shape), requires_grad=requires_grad)
        return unconditional_embeddings


    def _update_conditions(self, cond: Tensor, merge_func: Optional[Callable], batch_size: int) -> Dict[str, Tensor]:
        """
        Merge provided conditions with learned "unconditional" embeddings.

        Args:
            cond (Tensor): conditional embeddings
            merge_func(Callable): function defining how to merge the conditional and unconditional embedding.
            batch_size (int): batch size for the output embeddings
        """
        embedding = dict()
        # Pick the correct unconditional embedding for train or eval
        uncond = self.train_unconditional_embedding if self.training else self.eval_unconditional_embedding
        embedding = merge_func(cond, uncond) if merge_func else cond

        return embedding

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        b = x.shape[0]
        if self.training:
            # Classifier free guidance during training
            # Dropout randomly drops out conditional inputs based on self.p for learned unconditional ones
            def dropout_func(cond, uncond):
                return torch.where(torch.rand(b, *([1] * (len(cond.shape) - 1)), device=x.device) < self.p, uncond, cond)
            embedding = self._update_conditions(conditional_inputs, dropout_func, b)
            return self.model(x, timestep, embedding)
        elif self.guidance == 0 or conditional_inputs is not None:
            # If guidance is 0 or there are no conditional inputs to guide, then run inference
            # with no guidance. We still update conditions incase there are conditional inputs
            # and guidance is set to 0.
            embedding = self._update_conditions(conditional_inputs, None, b)
            return self.model(x, timestep, embedding)
        else:
            # Classifier free guidance during inference
            # Cat concatenates the conditional and unconditional input to compute both model outputs
            def cat_func(cond, uncond):
                return torch.cat([cond, uncond.expand_as(cond)], dim=0)
            embedding = self._update_conditions(conditional_inputs, cat_func, 2 * b)

            # Duplicate x and t to perform both conditional and unconditional generation
            x_ = torch.cat([x, x], dim=0)
            t_ = torch.cat([timestep, timestep], dim=0)
            output = self.model(x_, t_, embedding)
            cond, uncond = torch.chunk(output.prediction, 2, dim=0)
            # Combine conditional and unconditional generation
            output.prediction = (1 + self.guidance) * cond - self.guidance * uncond

            # variance_value is duplicated, so deduplicating
            if output.variance_value is not None:
                output.variance_value, _ = torch.chunk(output.variance_value, 2, dim=0)
            return output
