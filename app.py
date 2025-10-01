import streamlit as st
import numpy as np
import time
from PIL import Image
import tflite_runtime.interpreter as tflite

# --- Load TFLite model ---
@st.cache_resource
def load_model(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: Image.Image, input_shape):
    # Convert to RGB, resize, normalize
    image = image.convert("RGB").resize((input_shape[1], input_shape[2]))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(interpreter, image: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference + timing
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # ms

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, inference_time


# --- Streamlit UI ---
st.title("🧪 Malaria Detection Demo")
st.write("Upload a blood smear image to classify it using a TFLite model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    interpreter = load_model("assets/model.tflite")
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Preprocess
    img_array = preprocess_image(image, input_shape)

    # Predict
    preds, inference_time = predict(interpreter, img_array)

    # Display results
    st.subheader("🔎 Prediction")
    st.write(f"Raw output: {preds}")
    st.write(f"⏱ Inference time: {inference_time:.2f} ms")