import os
from abc import ABCMeta, abstractmethod
from enum import StrEnum
from pathlib import Path

import pandas as pd


class DATASETS(StrEnum):
    MTG_JAMENDO = "mtg-jamendo"


class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_audio_paths(self) -> pd.Series:
        pass

    @abstractmethod
    def get_labels(self) -> pd.Series:
        pass

    @abstractmethod
    def get_urls(self) -> pd.Series:
        pass


def get_dataset(dataset_name: str) -> Dataset:
    if dataset_name == DATASETS.MTG_JAMENDO:
        from .mtg_jamendo import MtgJamendoDataset

        metadata_file = os.getenv("MTGJAMENDO_METADATA_PATH")
        if metadata_file is None:
            raise RuntimeError("MTGJAMENDO_METADATA_PATH not set")
        metadata_file_path = Path(metadata_file)  # required

        root_dir = os.getenv("MTGJAMENDO_AUDIO_DIR")
        root_dir_path = Path(root_dir) if root_dir else None  # optional

        return MtgJamendoDataset(metadata_file_path, root_dir_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
