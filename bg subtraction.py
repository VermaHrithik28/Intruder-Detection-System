import cv2

def main():
    # Load video
    video = cv2.VideoCapture(r"intruder.mp4")

    # Create background subtractor (mixture of gaussian 2)
    back_sub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Display the foreground mask
        cv2.imshow("Foreground Mask", fg_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
