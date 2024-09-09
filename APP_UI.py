import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tempfile
from PIL import Image


def detect(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img


def main():
    st.title("Face Detection and Blurring")

    # Sidebar selection for mode
    option = st.sidebar.selectbox(
        "Select input type:",
        ("Image", "Video")
    )

    # Face detection model initialization
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

        if option == "Image":
            st.header("Upload Image")

            uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

            if uploaded_image is not None:
                image = Image.open(uploaded_image)


                img = np.array(image)

                processed_img = detect(img, face_detection)

                st.image(image, caption='Original Image', use_column_width=True)
                st.image(processed_img, caption='Processed Image (Faces Blurred)', use_column_width=True)

        elif option == "Video":
            st.sidebar.header("Upload Video")

            uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name

                cap = cv2.VideoCapture(video_path)

                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = detect(frame, face_detection)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    stframe.image(frame, channels='RGB', use_column_width=True)

                cap.release()


if __name__ == "__main__":
    main()
