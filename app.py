import streamlit as st
from predict import predict_food
import os

st.set_page_config(page_title="Food Calorie Estimator")

st.title("🍔 Food Calorie Estimator")
st.write("Upload an image to detect food and estimate calories")

uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_path = "temp.jpg"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Loading spinner
    with st.spinner("🔍 Analyzing image... Please wait"):
        food, calories, _ = predict_food(file_path)  

    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    st.success(f"🍽 Food: {food}")
    st.info(f"🔥 Calories: {calories} kcal")

    if os.path.exists(file_path):
        os.remove(file_path)