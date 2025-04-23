# IMPORT REQUIRED LIBRARIES

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# LOAD THE MODEL AND PREPROCESSING TOOLS

# Load the trained LSTM model
model = load_model('airline_sentiment_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the max sequence length used during training
with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)
    
    
# CREATE THE STREAMLIT INTERFACE

# Add Navigation in the Sidebar
st.sidebar.title("✈️ Airline Review Sentiment Analysis App")
page = st.sidebar.radio("Go to:", ["Single Review Analysis", "Bulk Review Analysis"])

# Single Review Analysis
if page == "Single Review Analysis":

    # Title and description
    st.title("Single Review Sentiment Analysis")
    st.markdown("Enter a passenger review below to predict whether the sentiment is **Positive** or **Negative**.")

    # Text input box
    user_input = st.text_area("📝 Enter your review here:", height=150)

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            # Preprocess input using tokenizer
            sequence = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

            # Make prediction
            prediction = model.predict(padded)[0][0]

            # Interpret prediction
            sentiment = "👍 Positive" if prediction >= 0.5 else "👎 Negative"
            confidence = round(float(prediction) * 100 if prediction >= 0.5 else (1 - float(prediction)) * 100, 2)

            # Show result
            st.success(f"**Sentiment:** {sentiment}")
            st.info(f"**Confidence:** {confidence}%")

# Bulk Review Analysis
elif page == "Bulk Review Analysis":

    # Title and description
    st.title("📁 Bulk Review Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file with a column named 'review':", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'review' not in df.columns:
                st.error("CSV must contain a column named 'review'")
            else:
                st.success(f"✅ {len(df)} reviews loaded successfully.")

                # Preprocess reviews
                sequences = tokenizer.texts_to_sequences(df['review'].astype(str))
                padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
                preds = model.predict(padded)

                # Convert predictions to readable format
                df['sentiment'] = ['Positive' if p >= 0.5 else 'Negative' for p in preds]
                df['confidence'] = [round(float(p) * 100 if p >= 0.5 else (1 - float(p)) * 100, 2) for p in preds]

                # Filter option
                st.markdown("### 🔎 Filter Reviews")
                filter_option = st.selectbox("Filter by sentiment:", ["All", "Positive", "Negative"])
                filtered_df = df if filter_option == "All" else df[df['sentiment'] == filter_option]

                st.dataframe(filtered_df)
                
                # Downloadable CSV
                from io import BytesIO
                towrite = BytesIO()
                filtered_df.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("⬇️ Download Result CSV", towrite, file_name='predicted_reviews.csv', mime='text/csv')

                # Sentiment Distribution Bar Chart
                st.markdown("## 📊 Sentiment Distribution")

                sentiment_counts = df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                fig = px.bar(
                    sentiment_counts,
                    x='Sentiment',
                    y='Count',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': 'green',
                        'Negative': 'red'
                    },
                    text='Count',
                    title='Number of Reviews by Sentiment',
                    labels={'Sentiment': 'Sentiment Type', 'Count': 'Number of Reviews'},
                    height=400
                )

                fig.update_traces(textposition='outside')
                fig.update_layout(
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    xaxis_title='Sentiment',
                    yaxis_title='Review Count',
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Word Cloud
                st.markdown("## ☁️ Word Clouds by Sentiment")

                # Separate positive and negative reviews
                positive_text = " ".join(df[df['sentiment'] == 'Positive']['review'].astype(str))
                negative_text = " ".join(df[df['sentiment'] == 'Negative']['review'].astype(str))

                # Generate and display positive word cloud
                if positive_text.strip():
                    st.markdown("### 🟢 Positive Reviews")
                    pos_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(pos_wc, interpolation='bilinear')
                    ax1.axis('off')
                    st.pyplot(fig1)
                else:
                    st.info("No positive reviews to display.")

                # Generate and display negative word cloud
                if negative_text.strip():
                    st.markdown("### 🔴 Negative Reviews")
                    neg_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(neg_wc, interpolation='bilinear')
                    ax2.axis('off')
                    st.pyplot(fig2)
                else:
                    st.info("No negative reviews to display.")
        except Exception as e:
            st.error(f"Error processing file: {e}")




