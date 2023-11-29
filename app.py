import streamlit as st
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

st.set_page_config(
    page_title="Sentiment Analysis", page_icon="🔥", layout="centered"
)


# # Sidebar
# st.sidebar.title("Sentiment Analysis App")
# st.sidebar.subheader("\n\n\n\nCreator Profile")

# # Creator information
# creator_name = "Bryan Melvin Jeffryson"
# creator_study = "Bina University, Indonesia"
# creator_image_path = "imageprofile.jpeg"

# # Display creator profile
# st.sidebar.image(creator_image_path, caption=creator_name, width=150)
# st.sidebar.write(f"**Name:** {creator_name}")
# st.sidebar.write(f"**Study at:** {creator_study}")


st.title("Sentiment Analysis of Product Review")
# Change text_area to text_input
review = st.text_area("Enter your product review here:", height=150)

# Add a button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if review:
        vect_path = 'vectorizer_app.pkl'
        vect = joblib.load(vect_path)

        text = np.array([review], dtype=object)
        sample = vect.transform(text)

        model_path = 'model_app.pkl'
        model = joblib.load(model_path)

        with st.spinner("Analyzing..."):
            time.sleep(2)
            predicted = model.predict(sample)

        if predicted == "__label__1":
            st.error("❌ This review is **Negative**.")
        else:
            st.success("✅ This review is **Positive**.")
    else:
        st.warning("Please enter a product review.")
st.write(
    """
    ## How it Works:
    - Enter your product review in the text area above.
    - Click the button to analyze the sentiment.
    - The system will predict whether the review is positive or negative based on a pre-trained model.

    ## About the Model:
    This sentiment analysis model was trained on a dataset of product reviews using a Support Vector Machine (SVM).
    """
)

st.markdown(
    """
    ---\n\n\n\n
    Made by Bryan Melvin Jeffryson
    """
)
