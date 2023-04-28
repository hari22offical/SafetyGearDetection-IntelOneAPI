import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import tensorflow as tf

# loading the DenseNet model
model = tf.keras.models.load_model("C:/Users/jeyasri/Downloads/densenet.h5")

# class labels
safety_gear_labels = ["helmet", "vest", "person", "no-vest", "no-helmet"]

# function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# function to preprocess the video frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# function to perform object detection on image
def detect_objects_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = safety_gear_labels[class_index]
    confidence = prediction[0][class_index] * 100
    return class_label, confidence

# function to perform object detection on video frames
def detect_objects_video(frame):
    frame = preprocess_frame(frame)
    prediction = model.predict(frame)
    class_index = np.argmax(prediction)
    class_label = safety_gear_labels[class_index]
    confidence = prediction[0][class_index] * 100
    return class_label, confidence

# streamlit app
def app():
    st.title("Safety Gear Object Detection for Construction Workers")
    st.write("Upload an image or a video, or start webcam streaming to detect safety gear objects.")

    # Sidebar for selecting input type
    input_type = st.sidebar.selectbox("Select Input Type", options=["Image", "Video", "Webcam"])

    if input_type == "Image":
        # Upload and detect image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect"):
                class_label, confidence = detect_objects_image(image)
                st.write("Detected Class Label:", class_label)
                st.write("Confidence:", round(confidence, 2), "%")

    elif input_type == "Video":
        # Upload and detect video
        uploaded_file = st.file_uploader("Choose a video", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_name = temp_file.name

            cap = cv2.VideoCapture(temp_file_name)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    class_label, confidence = detect_objects_video(frame)
                    cv2.putText(frame, f"{class_label} {confidence:.2f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.image(frame, channels="RGB", use_column_width=True)

                if st.button("Detect"):
                    class_label, confidence = detect_objects_video(frame)
                    st.write("Detected Class Label:", class_label)
                    st.write("Confidence:", round(confidence, 2), "%")
                if st.button("Stop"):
                    cap.release()
                    break

    elif input_type == "Webcam":
        # Detect objects from webcam stream
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                class_label, confidence = detect_objects_video(frame)
                cv2.putText(frame, f"{class_label} {confidence:.2f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.image(frame, channels="RGB", use_column_width=True)

                if st.button("Stop"):
                    cap.release()
                    break

if __name__ == '__main__':
    app()
