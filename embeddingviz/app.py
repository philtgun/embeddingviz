import os

import dotenv
import streamlit as st
from audio_plot import audio_plot
from umap import UMAP

from embeddingviz.datasets import DATASETS, get_dataset
from embeddingviz.models import MODELS, get_model
from embeddingviz.processor import get_processor

dotenv.load_dotenv()

api_token = os.getenv("HUGGINGFACE_TOKEN")

st.set_page_config(layout="wide")

with st.sidebar:
    dataset_select = st.selectbox("Dataset", list(DATASETS))
    model_select = st.selectbox("Select a model", list(MODELS))

if dataset_select is not None and model_select is not None:
    dataset = get_dataset(dataset_select)
    model = get_model(model_select)

    dataset_paths = dataset.get_audio_paths()[:5]

    processor = get_processor(model)
    with st.spinner("Processing audio..."):  # TODO offline
        embeddings = processor.process_audio_batch(dataset_paths.to_list(), model)

    umap = UMAP(n_components=2)
    with st.spinner("Projecting onto 2D..."):  # TODO offline
        projected_embeddings = umap.fit_transform(embeddings)

    labels = dataset.get_labels()[:5]
    urls = dataset.get_urls()[:5]

    audio_plot(projected_embeddings, urls, labels)
