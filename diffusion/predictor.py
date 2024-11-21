from abc import abstractmethod
from typing import Callable, Optional, Protocol

from torch import Tensor

from .schedule import DiscreteGaussianSchedule


class Predictor(Protocol):
    """Helper class to help predict various parts of the diffusion process. Different
    implementations of each method are needed depending on what the model itself was
    trained to predict.
    """

    schedule: DiscreteGaussianSchedule
    clamp_func: Optional[Callable]

    @abstractmethod
    def predict_x0(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Predict x0

        Args:
            prediction (Tensor): model prediction
            xt (Tensor): noised data to step t
            t (Tensor): int diffusion step for xt
        """

    @abstractmethod
    def predict_noise(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Predict noise

        Args:
            prediction (Tensor): model prediction
            xt (Tensor): noised data to step t
            t (Tensor): int diffusion step for xt
        """


class NoisePredictor(Predictor):
    """Given a model that's trained to predict diffusion noise and corresponding schedule,
        this class computes the predicted noise and x0 at step t.

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        clamp_func (Callable): function to clamp prediction values
    """

    def __init__(
        self, schedule: DiscreteGaussianSchedule, clamp_func: Optional[Callable] = None
    ):
        self.clamp_func = clamp_func
        self.schedule = schedule

    def predict_x0(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        shape, dtype = xt.shape, xt.dtype
        x_coef = self.schedule("sqrt_recip_alphas_cumprod", t, shape)
        e_coef = self.schedule("sqrt_recip_alphas_cumprod_minus_one", t, shape)
        x0 = x_coef * xt - e_coef * prediction
        if self.clamp_func is not None:
            x0 = self.clamp_func(x0)
        return x0.to(dtype)

    def predict_noise(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        return prediction
