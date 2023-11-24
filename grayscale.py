import cv2

def main():
    # Load video
    video = cv2.VideoCapture(r"intruder.mp4")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow("Grayscale", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
