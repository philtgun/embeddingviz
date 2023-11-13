from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from .datasets import DATASETS, get_dataset
from .models import MODELS, get_model
from .processor import get_processor


def process_offline(dataset_name: DATASETS, model_name: MODELS, device: str) -> None:
    dataset = get_dataset(dataset_name)
    model = get_model(model_name)

    processor = get_processor(model, device)
    processor.process_audio_batch(dataset.get_audio_paths().to_list(), model)
    # TODO: batched, save intermediate full embeddings, dimension reduction


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=DATASETS, choices=list(DATASETS))
    parser.add_argument("model", type=MODELS, choices=list(MODELS))
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
