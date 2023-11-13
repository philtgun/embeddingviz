import torch
from transformers import ClapModel, ClapProcessor

from . import Model


class GenericClapModel(Model):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(sample_rate=48_000, **kwargs)
        model_name = f"laion/{model_name}"
        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)

    def get_text_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt")
        return self.model.get_text_features(**inputs)

    def get_audio_embeddings_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        audios = [waveform.numpy() for waveform in waveforms]  # huggingface ClapProcessor expects numpy
        inputs = self.processor(audios=audios, return_tensors="pt")
        return self.model.get_audio_features(**inputs)
