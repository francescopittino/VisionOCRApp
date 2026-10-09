import streamlit as st
import json
import base64
from openai import OpenAI

st.set_page_config(page_title="VisionOCR", layout="wide")

with open("./cred.json", "r") as f:
    data = json.load(f)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=data["key"]
)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

@st.cache_data(ttl=3600)
def get_openrouter_models():
    model_list = client.models.list()
    valid_models = []
    for model in model_list:
        model_dict = model.model_dump()
        architecture = model_dict["architecture"]
        input_modalities = architecture["input_modalities"]
        if "image" in input_modalities:
            valid_models.append(model.id)

    return sorted(valid_models)

#Sidebar
with st.sidebar:

    models = get_openrouter_models()

    llm_model = st.selectbox(
        label = "Select LLM Vision Model",
        options = models,
    )

    output_format = st.radio(
        "Formato di Output",
        ["Plain Text", "Markdown Strutturato", "JSON"]
    )

uploaded_file = st.file_uploader("Carica un'immagine o un documento (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Immagine Caricata")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("Risultato Trascrizione")

        if st.button("Esegui Trascrizione", type="primary"):
            with st.spinner("Analisi visiva dell'immagine in corso..."):
                base64_img = encode_image(uploaded_file)
                mime_type = uploaded_file.type

                system_prompt = "You are an advanced OCR system. Given an image that contains text, return a transcription of it. If parts of the text are not readable/are ambiguous, mark them as such and do not try to guess."
                user_prompt = f"Give an accurate transcription. For the output, use the {output_format} format. Output only the transcription, nothing else."

                try:
                    response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_img}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.2
                    )

                    extracted_text = response.choices[0].message.content
                    st.success("Trascrizione completata!")

                    if output_format == "JSON":
                        st.code(extracted_text, language="json")
                    else:
                        st.markdown(extracted_text)

                    st.download_button(
                        label="Scarica Trascrizione(.txt)",
                        data=extracted_text,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Errore durante l'elaborazione: {e}")