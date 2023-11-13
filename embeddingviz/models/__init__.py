from abc import ABCMeta, abstractmethod
from enum import StrEnum

import torch


class MODELS(StrEnum):
    CLAP_HTSAT_UNFUSED = "clap-htsat-unfused"
    LARGER_CLAP_MUSIC = "larger_clap_music"


class Model(metaclass=ABCMeta):
    def __init__(self, sample_rate: int, device: str = "cpu") -> None:
        self.sample_rate = sample_rate
        self.device = device

    @abstractmethod
    def get_text_embeddings(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_audio_embeddings_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        pass


def get_model(model_name: str) -> Model:
    if model_name in [MODELS.CLAP_HTSAT_UNFUSED, MODELS.LARGER_CLAP_MUSIC]:
        from .clap import GenericClapModel

        return GenericClapModel(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
