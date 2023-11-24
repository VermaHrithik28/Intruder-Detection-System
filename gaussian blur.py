import cv2

def apply_gaussian_filter(video_path, kernel_size=(5, 5), sigma=0):
    # Load video
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Apply Gaussian filter
        filtered_frame = cv2.GaussianBlur(frame, kernel_size, sigma)

        # Display the filtered frame
        cv2.imshow("Filtered Frame", filtered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video object
    video.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r"intruder.mp4"
apply_gaussian_filter(video_path)
