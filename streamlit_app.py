import streamlit as st
import os
import json
import numpy as np
import requests
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import cohere
from google import genai

# Load env variables
load_dotenv()

# API keys
cohere_api_key = os.getenv("coher_api_key")
google_api_key = os.getenv("google_api_key")

# Clients
co = cohere.ClientV2(cohere_api_key)
genai_client = genai.Client(api_key=google_api_key)

# Constants
img_folder = "images"
os.makedirs(img_folder, exist_ok=True)
data_file = "embeddings.json"
max_pixels = 1568 * 1568

# Helper: Resize and base64 encode image
def resize_image(pil_image):
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(pil_image, format="PNG"):
    resize_image(pil_image)
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=format)
        img_buffer.seek(0)
        return "data:image/{};base64,".format(format.lower()) + base64.b64encode(img_buffer.read()).decode("utf-8")

# Helper: Load or save embeddings
def load_data():
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(data_file, "w") as f:
        json.dump(data, f)

# Streamlit pages
st.set_page_config(page_title="Image QA App", layout="centered")
page = st.sidebar.radio("Navigation", ["Upload Image", "Ask Question"])

if page == "Upload Image":
    st.title("Upload Image(s) and Store Embedding")
    uploaded_files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        data = load_data()
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

                # Save locally
                img_path = os.path.join(img_folder, uploaded_file.name)
                image.save(img_path)

                # Base64 encode and embed
                base64_img = base64_from_image(image, format=image.format or "PNG")
                input_doc = {"content": [{"type": "image", "image": base64_img}]}

                api_response = co.embed(
                    model="embed-v4.0",
                    input_type="search_document",
                    embedding_types=["float"],
                    inputs=[input_doc],
                )
                emb = api_response.embeddings.float[0]

                # Store embedding
                data[img_path] = emb
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")

        save_data(data)
        st.success("All embeddings saved successfully!")

elif page == "Ask Question":
    st.title("Ask a Question Based on Image Content")
    question = st.text_input("Enter your question")
    
    if question:
        data = load_data()
        if not data:
            st.warning("No images found. Please upload first.")
        else:
            # Embed question
            api_response = co.embed(
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
                texts=[question],
            )
            q_emb = np.array(api_response.embeddings.float[0])

            # Find best match
            img_paths = list(data.keys())
            img_embeddings = np.array([data[p] for p in img_paths])
            scores = np.dot(q_emb, img_embeddings.T)
            top_idx = np.argmax(scores)
            best_img_path = img_paths[top_idx]

            st.image(best_img_path, caption="Most Relevant Image")

            # Answer with Gemini
            prompt = [
                f"""Answer the question based on the following image.\nDon't use markdown.\nPlease provide enough context.\n\nQuestion: {question} and only provide the exact answer based on question and descript more beautiful format user understandable way ,don't explain other attributes in image only relvant information explain based on question""",
                Image.open(best_img_path),
            ]

            response = genai_client.models.generate_content(
                model="models/gemini-2.5-flash-preview-04-17",
                contents=prompt
            )

            st.subheader("Answer")
            st.write(response.text)
