import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .models import Model


class Processor:
    def __init__(self, sample_rate: int, start_time: float, duration: float, device: str) -> None:
        self.sample_rate = sample_rate
        self.device = device

        self.start_time = start_time
        self.duration = duration

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        waveform = waveform.mean(dim=0)
        return waveform, sample_rate

    def _trim_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        start_sample = int(self.start_time * sample_rate)
        end_sample = start_sample + int(self.duration * sample_rate)
        return waveform[start_sample:end_sample]

    def _resample_audio(self, waveform: torch.Tensor, sample_rate: int, new_sample_rate: int):
        return torchaudio.transforms.Resample(sample_rate, new_sample_rate).to(self.device)(waveform)

    def _load_and_resample(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = self._load_audio(audio_path)
        waveform = self._trim_audio(waveform, sample_rate)
        return self._resample_audio(waveform, sample_rate, self.sample_rate)

    def _load_and_resample_batch(self, audio_paths: list[Path]) -> list[torch.Tensor]:
        waveforms = [self._load_and_resample(audio_path) for audio_path in audio_paths]
        return waveforms

    def process_audio_batch(self, audio_paths: list[Path], model: Model) -> np.ndarray:
        waveforms = self._load_and_resample_batch(audio_paths)
        embeddings = model.get_audio_embeddings_batch(waveforms)
        return embeddings.detach().numpy()


def get_processor(model: Model, device: str = "cpu") -> Processor:
    # will raise errors if not set
    start = os.environ["SECONDS_START"]
    duration = os.environ["SECONDS_DURATION"]

    return Processor(model.sample_rate, float(start), float(duration), device)
