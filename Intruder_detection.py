import cv2
import numpy as np

def main():
    # Load video
    video = cv2.VideoCapture("intruder.mp4")

    # Get FPS value from video capture object
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second:", fps)

    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    frame_number = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Grayscale and noise removal
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background subtraction
        fg_mask = back_sub.apply(blurred)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Object detection and tracking
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Filter small contours
            if cv2.contourArea(cnt) < 1000:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Check if the object is in the main zone
            if is_in_main_zone(x, w, frame.shape[1]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Intruder in main zone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Intruder detected in {frame_number}")

        # Display the frame
        cv2.imshow("Intruder Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

def is_in_main_zone(x, w, frame_width):
    # Define zone boundaries based on frame width
    entrance_zone = frame_width * 0.3
    main_zone = frame_width * 0.7

    object_center = x + w / 2

    return entrance_zone < object_center < main_zone 

if __name__ == "__main__":
    main()
