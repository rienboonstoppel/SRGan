import torch
import numpy as np
import scipy.ndimage as ndi

from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio.transforms.augmentation.intensity.random_gamma import Gamma
# from torchio.transforms.augmentation.intensity.random_blur import Blur
from torchio.transforms.augmentation.intensity.random_bias_field import BiasField

from torchio.typing import TypeRangeFloat
from torchio.utils import to_tuple
from collections import defaultdict
from typing import Union, Tuple, Dict, List
from torchio.typing import TypeSextetFloat, TypeTripletFloat, TypeData


class Normalize(IntensityTransform):
    def __init__(
            self,
            std,
            mean=0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.std = std
        self.mean = mean

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in self.get_images_dict(subject).items():
            self.apply_normalization(subject, image_name)
        return subject

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
    ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            self.std,
            self.mean,
        )
        image.set_data(standardized)

    @staticmethod
    def znorm(tensor: torch.Tensor, std, mean) -> torch.Tensor:
        tensor = tensor.clone().float()
        tensor -= mean
        tensor /= std
        return tensor


class RandomIntensity(RandomTransform, IntensityTransform):
    def __init__(
            self,
            intensity_diff: TypeRangeFloat = (-0.3, 0.3),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.intensity_diff_range = self._parse_range(intensity_diff, 'intensity_diff')

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        intensity = self.get_params(self.intensity_diff_range)
        for name, image in self.get_images_dict(subject).items():
            intensities = [
                intensity
                for _ in image.data
            ]
            arguments['intensity'][name] = intensities

        transform = Intensity(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, intensity_diff_range: Tuple[float, float]) -> float:
        intensity = self.sample_uniform(*intensity_diff_range).item()
        return intensity


class Intensity(IntensityTransform):
    def __init__(
            self,
            intensity: float,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.intensity = intensity
        self.args_names = ('intensity',)
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        intensity = self.intensity
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                intensity = self.intensity[name]
            intensities = to_tuple(intensity, length=len(image.data))
            transformed_tensors = []
            image.set_data(image.data.float())
            for intensity, tensor in zip(intensities, image.data):
                if self.invert_transform:
                    transformed_tensor = tensor.clone()
                    correction = -intensity
                    transformed_tensor[transformed_tensor.nonzero(as_tuple=True)] += correction
                else:
                    transformed_tensor = tensor.clone()
                    transformed_tensor[transformed_tensor.nonzero(as_tuple=True)] += intensity
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


class RandomGamma(RandomTransform, IntensityTransform):
    def __init__(
            self,
            log_gamma: TypeRangeFloat = (-0.3, 0.3),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.log_gamma_range = self._parse_range(log_gamma, 'log_gamma')

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        gamma = self.get_params(self.log_gamma_range)

        for name, image in self.get_images_dict(subject).items():
            gammas = [ gamma
                for _ in image.data
            ]
            arguments['gamma'][name] = gammas
        transform = Gamma(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, log_gamma_range: Tuple[float, float]) -> float:
        gamma = self.sample_uniform(*log_gamma_range).exp().item()
        return gamma


def _parse_order(order):
    if not isinstance(order, int):
        raise TypeError(f'Order must be an int, not {type(order)}')
    if order < 0:
        raise ValueError(f'Order must be a positive int, not {order}')
    return order


class RandomBiasField(RandomTransform, IntensityTransform):
    def __init__(
            self,
            coefficients: Union[float, Tuple[float, float]] = 0.5,
            order: int = 3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.coefficients_range = self._parse_range(
            coefficients, 'coefficients_range',
        )
        self.order = _parse_order(order)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        coefficients = self.get_params(
            self.order,
            self.coefficients_range,
        )
        for image_name in self.get_images_dict(subject):
            arguments['coefficients'][image_name] = coefficients
            arguments['order'][image_name] = self.order
        transform = BiasField(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(
            self,
            order: int,
            coefficients_range: Tuple[float, float],
    ) -> List[float]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = self.sample_uniform(*coefficients_range)
                    random_coefficients.append(number.item())
        return random_coefficients


class RandomBlur(RandomTransform, IntensityTransform):
    def __init__(
            self,
            std: TypeRangeFloat = (0, 2),
            **kwargs
    ):
        super().__init__(**kwargs)
        # self.std_range = self.parse_params(std, None, 'std', min_constraint=0, make_ranges=True)[:2]
        self.std_range = self._parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments: Dict[str, dict] = defaultdict(dict)
        std = self.get_params(self.std_range)

        arguments['std']['LR'] = std
        arguments['std']['HR'] = 0
        #         for name in self.get_images_dict(subject):
        #             std = self.get_params(self.std_ranges)
        #             arguments['std'][name] = std

        # if 'HR' in self.get_images_dict(subject):
        #     arguments['std']['HR'] = 0

        transform = Blur(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(self, std_range: Tuple[float, float]) -> float:
        std = self.sample_uniform(*std_range).item()
        return std


class Blur(IntensityTransform):
    def __init__(
            self,
            std: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.std = std
        self.args_names = ['std']

    def apply_transform(self, subject: Subject) -> Subject:
        stds = self.std
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                assert isinstance(self.std, dict)
                stds = self.std[name]
            repets = image.num_channels, 1
            stds_channels: np.ndarray
            stds_channels = np.tile(stds, repets)  # type: ignore[arg-type]
            transformed_tensors = []
            for std, channel in zip(stds_channels, image.data):
                transformed_tensor = blur(
                    channel,
                    std,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


def blur(
        data: TypeData,
        std: TypeTripletFloat,
) -> torch.Tensor:
    assert data.ndim == 3
    blurred = ndi.gaussian_filter(data, (std, std, 0))
    tensor = torch.as_tensor(blurred)
    return tensor
