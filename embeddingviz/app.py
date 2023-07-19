import os
from time import sleep

import dotenv
import requests
import streamlit as st
from transformers import ClapModel, ClapProcessor

dotenv.load_dotenv()

api_token = os.getenv("HUGGINGFACE_TOKEN")


model_select = st.selectbox("Select a model", ["", "CLAP", "Other"], index=0)

if model_select == "CLAP":
    model_name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)
    inputs = processor(text="metalcore with female vocals", return_tensors="pt")
    embdeddings = model.get_text_features(**inputs)
    st.write(embdeddings)
    st.write(len(embdeddings[0]))


# Trying api, doesn't work :(
if st.button("Test HuggingFace API"):
    retry = True
    while retry:
        result = requests.post(
            url="https://api-inference.huggingface.co/models/laion/clap-htsat-unfused",
            headers={"Authorization": f"Bearer {api_token}"},
            json={"inputs": [["metalcore with female vocals"]]},
        )
        if result.status_code == 503:
            wait_time = result.json()["estimated_time"] + 1.0
            with st.spinner(f"Waiting {wait_time:.0f} seconds for the model to load..."):
                sleep(wait_time)
            retry = True
        else:
            retry = False

    st.write(result.status_code, result.json())
