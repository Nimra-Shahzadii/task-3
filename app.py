import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("fake_news_lstm.keras")


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 200  

def predict_fake_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")

    prediction = model.predict(padded_seq)[0][0]

    label = "FAKE NEWS" if prediction > 0.5 else "REAL NEWS"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

    return f"{label} ({confidence}% confidence)"

iface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=5, placeholder="Enter a news article..."),
    outputs=gr.Label(),
    title="📰 Fake News Detector",
    description="Enter a news article to check if it's real or fake. The model is trained on a fake news dataset using LSTM.",
    theme="huggingface",
)

iface.launch()