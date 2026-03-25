import streamlit as st
from predict import predict_food

st.set_page_config(page_title="Food Calorie Estimator")

st.title("🍔 Food Calorie Estimator")
st.write("Upload an image to detect food and estimate calories")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    food, calories, confidence = predict_food("temp.jpg")

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    st.success(f"🍽 Food: {food}")
    st.info(f"🔥 Calories: {calories} kcal")
    st.write(f"Confidence: {confidence:.2f}")