import csv
import os
from pathlib import Path

import pandas as pd

from . import Dataset


class MtgJamendoDataset(Dataset):
    def __init__(self, metadata_file: Path, root_dir: Path | None) -> None:
        self.metadata, self.tags = self.read_tsv(metadata_file)
        self.root_dir = root_dir

        self.play_start = float(os.environ["SECONDS_START"])
        play_duration = float(os.environ["SECONDS_DURATION"])
        self.play_end = self.play_start + play_duration

    @staticmethod
    def read_tsv(tsv_file: Path) -> tuple[pd.DataFrame, dict[int, list[str]]]:
        metadata: dict[str, list] = {
            "track_id": [],
            "artist_id": [],
            "album_id": [],
            "path": [],
            "duration": [],
        }
        tags = {}
        with open(tsv_file) as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader, None)
            for row in reader:
                track_id = int(row[0].removeprefix("track_"))
                metadata["track_id"].append(track_id)
                metadata["artist_id"].append(int(row[1].removeprefix("artist_")))
                metadata["album_id"].append(int(row[2].removeprefix("album_")))
                metadata["path"].append(row[3])
                metadata["duration"].append(float(row[4]))

                tags[track_id] = row[5:]

        return pd.DataFrame(metadata), tags

    def __len__(self) -> int:
        return len(self.metadata)

    def _get_url(self, track_id: int) -> str:
        return f"https://prod-1.storage.jamendo.com/?trackid={track_id}&format=mp32#t={self.play_start},{self.play_end}"

    def get_urls(self) -> pd.Series:
        return self.metadata["track_id"].apply(self._get_url)

    def get_audio_paths(self) -> pd.Series:
        if self.root_dir is None:
            raise RuntimeError("Root directory not set")
        return self.metadata["path"].apply(lambda x: str(self.root_dir / x))

    def get_labels(self) -> pd.Series:
        return self.metadata["track_id"].apply(lambda x: f"Track {x} ({self.tags[x]})")
