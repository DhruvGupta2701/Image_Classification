import streamlit as st
import tensorflow
from PIL import Image
import numpy as np

st.title('Classifier')
classes = ['cats', 'dogs']

@st.cache_resource
def load_model():
    model = tensorflow.keras.models.load_model(
        r"C:\Users\Dhruv\OneDrive - iTachi World\Pictures\Documents\AI_ML_INTERN\DAY4\classifier.h5"
    )
    return model

model = load_model()

def preprocess(image: Image.Image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)
    return img

upload_f = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if upload_f:
    image = Image.open(upload_f)
    st.image(image, caption='Uploaded Image')
    processed = preprocess(image)
    pred = model.predict(processed)[0]  # Get 1D prediction array

    st.write(f"Raw prediction: {pred}")

    class_index = np.argmax(pred)

    if class_index < len(classes):
        pred_class = classes[class_index]
        st.success(f'Prediction: {pred_class}')
    else:
        st.error(f"Error: Predicted class index {class_index} exceeds defined class list.")
