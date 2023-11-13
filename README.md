# EmbeddigViz

TODO

## Env setup
```shell
python3.11 -m venv venv  # whichever way you installed python: bin, pyenv, conda, etc.
source venv/bin/activate.fish  # or .bash, .zsh, etc.
pip install --upgrade pip wheel  # upgrade pip and install wheel
make initdev  # install dev dependencies
```

## Environment variables
* `HUGGINGFACE_TOKEN`: HuggingFace API token
* `MTGJAMENDO_METADATA_PATH`: **UI** path to the metadata file (including `mtg-jamendo-dataset/data/splits/split-0/autotagging_top50tags-test.tsv`)
* `MTGJAMENDO_AUDIO_DIR`: **Process** path where the audio files are stored (should contain `00`, `01`, etc. directories inside)
* `SECONDS_START`: 60.0
* `SECONDS_DURATION`: 10.0

## Run
```shell
streamlit run embeddingviz/app.py
```

## Process audio
```

```
