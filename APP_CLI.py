import os
import argparse
import cv2
import mediapipe as mp


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
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img


def process_image(file_path, face_detection, output_dir):
    img = cv2.imread(file_path)

    if img is None:
        print(f"Image file {file_path} not found!")
        return

    img = detect(img, face_detection)

    # Save the processed image
    output_path = os.path.join(output_dir, 'output_image.png')
    cv2.imwrite(output_path, img)
    print(f"Processed image saved at {output_path}")
    return img



def process_video(file_path, face_detection, output_dir):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f"Video file {file_path} not found!")
        return

    ret, frame = cap.read()

    if not ret:
        print("Failed to read the video.")
        return

    output_video = cv2.VideoWriter(os.path.join(output_dir, 'output_video.mp4'),
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   25,
                                   (frame.shape[1], frame.shape[0]))

    while ret:
        frame = detect(frame, face_detection)
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()
    print(f"Processed video saved at {os.path.join(output_dir, 'output_video.mp4')}")



def process_webcam(face_detection):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    ret, frame = cap.read()
    while ret:
        frame = detect(frame, face_detection)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream closed.")



def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['image', 'video', 'webcam'], default='webcam', help="Mode: image, video, or webcam")
    parser.add_argument("--filePath", default=None, help="Path to the image or video file")
    args = parser.parse_args()

    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == "image":
            if args.filePath:
                process_image(args.filePath, face_detection, output_dir)
            else:
                print("Please provide the file path for the image using --filePath.")

        elif args.mode == "video":
            if args.filePath:
                process_video(args.filePath, face_detection, output_dir)
            else:
                print("Please provide the file path for the video using --filePath.")

        elif args.mode == "webcam":
            process_webcam(face_detection)


if __name__ == "__main__":
    main()
