import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import io
from tensorflow.keras.optimizers import Adam
#from streamlit_webrtc import webrtc_streamer

model_variants = {
    "Original Model (h5)": "models/model.h5",
    "TFLite Model": "models/model.tflite",
    "Pruned TFLite Model": "models/model_pruned.tflite",
    "Quantized TFLite Model": "models/model_quantized.tflite",
    "Pruned & Quantized TFLite Model": "models/model_pruned_quantized.tflite"
}

# Load a TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load a Keras model
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'Adam': Adam})

# Predict waste class using a Keras model
def predict_waste_class_keras(model, image):
    input_size = tuple(model.layers[0].input_shape[1:3])
    image_preprocessed = load_and_preprocess_image(image, input_size)
    output_data = model.predict(image_preprocessed[np.newaxis, ...])
    prediction = np.argmax(output_data)
    return prediction

def load_and_preprocess_image(image_path, target_size):
    # Load the image using PIL
    img = Image.open(image_path)
    
    # Resize the image to the target size
    img = img.resize(target_size, Image.ANTIALIAS)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Preprocess the image according to your model's requirements
    # This step depends on your specific model and data
    # For example, you may need to normalize the input data or convert it to grayscale
    # img_preprocessed = preprocess_input_data(img_array)  # Replace with your preprocessing function

    return img_array
# Predict waste class using a TFLite model

def predict_waste_class_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = tuple(input_details[0]['shape'][1:3])

    image_preprocessed = load_and_preprocess_image(image, input_size)

    interpreter.set_tensor(input_details[0]['index'], image_preprocessed[np.newaxis, ...].astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    return prediction

# Web app
st.title("Waste Classification")
st.header("Classify waste using your camera")

# Model selection
model_variant = st.sidebar.selectbox("Select Model Variant", list(model_variants.keys()))

# Load the selected model
if model_variants[model_variant].endswith(".h5"):
    model = load_keras_model(model_variants[model_variant])
    predict_fn = predict_waste_class_keras
else:
    interpreter = load_tflite_model(model_variants[model_variant])
    predict_fn = predict_waste_class_tflite

class_labels = ["cardboard", "glass", "metal", "paper", "plastic"]

webrtc_ctx = webrtc_streamer(key="snapshot")

if st.button("Capture Snapshot"):
    if webrtc_ctx.video_receiver:
        try:
            snapshot_image = webrtc_ctx.video_receiver.to_image()
            snapshot_image.save("snapshot.png")

            # Open the captured image
            image = Image.open("snapshot.png")
            st.image(image, caption="Snapshot", use_column_width=True)

            # Predict waste class
            if predict_fn == predict_waste_class_keras:
                prediction = predict_fn(model, image)
            else:
                prediction = predict_fn(interpreter, image)
                
            st.success(f"Predicted waste class: {class_labels[prediction]}")

        except Exception as e:
            st.error("Error: " + str(e))
    else:
        st.error("No video source available.")
