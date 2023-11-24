import cv2
import numpy as np

def draw_zones(frame, entrance_line, main_line):
    cv2.line(frame, entrance_line[0], entrance_line[1], (0, 255, 0), 2)
    cv2.line(frame, main_line[0], main_line[1], (255, 0, 0), 2)

def main():
    # Load the video (852 width of vd)
    video_path = r"intruder.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the zone lines
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    entrance_line = ((0, 300), (frame_width, 300))
    main_line = ((0, 80), (frame_width, 80))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw the zones on the grayscale frame
        draw_zones(gray, entrance_line, main_line)

        # Convert the grayscale frame back to BGR for display
        frame_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Display the frame
        cv2.imshow("Intruder Detection", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
