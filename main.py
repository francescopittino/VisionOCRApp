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
        "Output format",
        ["Plain Text", "Markdown", "JSON"]
    )

uploaded_file = st.file_uploader("Load an image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image uploaded")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("Result")

        if st.button("Elaborate image", type="primary"):
            with st.spinner("Visual analysis in progress..."):
                base64_img = encode_image(uploaded_file)
                mime_type = uploaded_file.type

                #Currently using Role-Prompting (zero-shot to limit token usage)
                system_prompt = """You are an OCR (Optical Character Recognition) system. 
                    Your sole task is to transcribe readable text from images.
                    
                    FOLLOW THESE RULES:
                    1. *Fidelity*: Transcribe the text exactly as it appears. Do not fix spelling mistakes, grammar errors, or typos present in the original image.
                    2. *Layout Preservation*: Maintain the original structure, paragraphs, line breaks, and indentation as much as possible. If there are tables or lists, preserve their formatting.
                    3. *Ambiguity Handling*: If a character, word, or section is illegible, blurry, or ambiguous, represent it explicitly using the marker [unreadable] or [illegible]. Never try to guess missing or unreadable text.
                    4. *No Conversational Filler*: Do not add introductory text, explanations, or conclusions. Return exclusively the extracted text content.
                    """
                
                if output_format == "JSON":
                    user_prompt = "Transcribe the text in this image and output it strictly as a valid JSON object with meaningful keys representing the document's structure"
                elif output_format == "Markdown":
                    user_prompt = "Transcribe the text preserving structural elements like headers, lists, and tables using Markdown syntax"
                else:
                    user_prompt = "Give an accurate plain text transcription maintaining paragraph breaks"

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
                        temperature=0.2 #low temp to reduce hallucinations, might want to lower it further
                    )

                    extracted_text = response.choices[0].message.content
                    st.success("Done!")

                    if output_format == "JSON":
                        st.code(extracted_text, language="json")
                    else:
                        st.markdown(extracted_text)

                    st.download_button(
                        label="Save(.txt)",
                        data=extracted_text,
                        file_name="transcription.txt", #To add: json and markdown
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
