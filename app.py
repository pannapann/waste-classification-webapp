import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import io
from tensorflow.keras.optimizers import Adam
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import os
import time
import tvm
from tvm import relay
from tvm.runtime import load_module
from tvm.contrib import graph_executor
import onnxruntime as rt

model_variants = {
    "Original Model (h5)": "models/model.h5",
    "TFLite Model": "models/model.tflite",
    "Pruned TFLite Model": "models/model_pruned.tflite",
    "Quantized TFLite Model": "models/model_quantized.tflite",
    "Pruned & Quantized TFLite Model": "models/model_pruned_quantized.tflite",
    "ONNX Model": "models/model.onnx",
    "TVM Model": "models/tvm_model"

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
    input_size = tuple(model.layers[0].input_shape[1:3]) or (224, 224)
    image_preprocessed = load_and_preprocess_image(image, input_size)
    output_data = model.predict(image_preprocessed[np.newaxis, ...])
    prediction = np.argmax(output_data)
    return prediction

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image using PIL
    img = Image.open(image_path)
    
    # Resize the image to the target size
    img = img.resize(target_size, Image.ANTIALIAS)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    img_array = img_array / 255.0

    return img_array

def load_and_preprocess_image_v2(image_path, target_size=(224, 224)):
    resized_image = Image.open(image_path).resize(target_size)
    img_data = np.asarray(resized_image).astype("float32")
    norm_img_data = img_data/255.0

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data


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

def load_tvm_model(model_path):
    loaded_lib = load_module(f"{model_path}/compiled_tvm_model.tar")

    with open(f"{model_path}/tvm_model_params.params", "rb") as f:
        loaded_params = bytearray(f.read())

    target = "llvm"  # Change this to the target you used during compilation
    loaded_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device(str(target), 0)))

    loaded_module.load_params(loaded_params)

    return loaded_module

def predict_waste_class_tvm(module, image):
    input_name = "args_0"
    input_shape = (1, 3, 224, 224)

    image_preprocessed = load_and_preprocess_image_v2(image, input_shape[2:])
    image_preprocessed = image_preprocessed[np.newaxis, ...].astype(np.float32)
    image_preprocessed = np.squeeze(image_preprocessed, axis=1)


    module.set_input(input_name, image_preprocessed)
    module.run()

    output_data = module.get_output(0).asnumpy()
    prediction = np.argmax(output_data)

    return prediction

def load_onnx_model(model_path):
    session = rt.InferenceSession(model_path)
    return session

def predict_waste_class_onnx(session, image):
    input_name = session.get_inputs()[0].name
    input_shape = (1, 3, 224, 224)

    image_preprocessed = load_and_preprocess_image_v2(image, input_shape[2:])
    image_preprocessed = image_preprocessed[np.newaxis, ...].astype(np.float32)
    image_preprocessed = np.squeeze(image_preprocessed, axis=1)


    output_data = session.run(None, {input_name: image_preprocessed})[0]
    prediction = np.argmax(output_data)

    return prediction




class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        self.last_frame = frame.to_ndarray(format="bgr24")
        return frame



# Web app
st.title("Waste Classification")
st.header("Classify waste using your camera")

# Model selection
model_variant = st.sidebar.selectbox("Select Model Variant", list(model_variants.keys()))

# Load the selected model
if model_variants[model_variant].endswith(".h5"):
    model = load_keras_model(model_variants[model_variant])
    predict_fn = predict_waste_class_keras
elif model_variants[model_variant].endswith(".tflite"):
    interpreter = load_tflite_model(model_variants[model_variant])
    predict_fn = predict_waste_class_tflite
elif model_variants[model_variant].endswith(".onnx"):
    session = load_onnx_model(model_variants[model_variant])
    predict_fn = predict_waste_class_onnx
else:
    module = load_tvm_model(model_variants[model_variant])
    predict_fn = predict_waste_class_tvm


class_labels = ["cardboard", "glass", "metal", "paper", "plastic"]


webrtc_ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

if st.button("Capture Snapshot"):
    if webrtc_ctx.video_transformer:
        try:
            snapshot_image_bgr = webrtc_ctx.video_transformer.last_frame
            snapshot_image_rgb = cv2.cvtColor(snapshot_image_bgr, cv2.COLOR_BGR2RGB)
            snapshot_image = Image.fromarray(snapshot_image_rgb)
            snapshot_image.save("snapshot.jpg")

            st.image(snapshot_image, caption="Snapshot", use_column_width=True)

            # Predict waste class
            if predict_fn == predict_waste_class_keras:
                start_time = time.time()
                prediction = predict_fn(model, "snapshot.jpg")
                inference_time = time.time() - start_time
            elif predict_fn == predict_waste_class_tflite:
                start_time = time.time()
                prediction = predict_fn(interpreter, "snapshot.jpg")
                inference_time = time.time() - start_time
            elif predict_fn == predict_waste_class_onnx:
                start_time = time.time()
                prediction = predict_fn(session, "snapshot.jpg")
                inference_time = time.time() - start_time
            else:
                start_time = time.time()
                prediction = predict_fn(module, "snapshot.jpg")
                inference_time = time.time() - start_time

                
            st.success(f"Predicted waste class: {class_labels[prediction]}")

            st.write(model_variant)

            st.write(f"Inference time: {inference_time:.2f} seconds")
            
            file_path = model_variants[model_variant]
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            st.write(f"Model size: {file_size_mb:.2f} MB")


        except Exception as e:
            st.error("Error: " + str(e))
    else:
        st.error("No video source available.")
