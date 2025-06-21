import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

st.set_page_config(
    page_title="IMDB Age Detector",
    page_icon="🎂",
    layout="centered",
)

@st.cache_resource
def load_age_model():
    model_path = "age_model.h5" 
    return load_model(model_path, compile=False)

def preprocess_image(uploaded_image):
    image = uploaded_image.convert("L")          
    image = image.resize((96, 96))                 
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1) 
    return np.expand_dims(image_array, axis=0)         

def predict_age(model, image_array):
    gender_pred, age_pred = model.predict(image_array)
    return int(np.round(age_pred[0][0]))

def main():
    st.markdown("<h1 style='text-align:center;'>IMDB Age Detector</h1>", unsafe_allow_html=True)

    model = load_age_model()

    uploaded_files = st.file_uploader(
        "Upload one or more face images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Detect Age", key="detect_age_button"):
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f"### Image {i+1}")
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

                    processed_image = preprocess_image(image)
                    predicted_age = predict_age(model, processed_image)

                    st.success(f"🎉 Predicted Age: {predicted_age} years")
        else:
            st.info("Please upload at least one image before clicking 'Detect Age'.")

    st.markdown("<hr><center>🚀 Powered by Vansh & IMDB Dataset</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
