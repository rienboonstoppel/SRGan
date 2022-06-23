import torch
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio.transforms.augmentation.intensity.random_gamma import Gamma
from torchio.typing import TypeRangeFloat
from torchio.utils import to_tuple
from collections import defaultdict
from typing import Tuple


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
                    correction = -intensity
                    transformed_tensor = tensor + correction
                else:
                    transformed_tensor = tensor + intensity
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