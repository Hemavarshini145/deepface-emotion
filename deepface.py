from deepface import DeepFace
import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        emotion = result[0]['dominant_emotion']

        # Put the detected emotion text on the frame
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    except Exception as e:
        # If face not detected or error occurs, just pass
        pass

    # Show the video frame
    cv2.imshow("DeepFace Emotion Detection", frame)

    # Break loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
