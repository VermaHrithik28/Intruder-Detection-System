import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the video file path
video_path = r'intruder.mp4'

# Create a video capture object
video = cv2.VideoCapture(video_path)

# Initialize the list to store the histograms and noise information
histograms = []
noises = []

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the frame
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histograms.append(histogram)
    
    # Apply a median filter to the frame to reduce noise
    filtered = cv2.medianBlur(gray, 5)
    
    # Calculate the difference between the original and filtered frame
    diff = cv2.absdiff(gray, filtered)
    
    # Apply thresholding to the difference image to detect noise
    threshold = 30
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Count the number of white pixels in the thresholded image
    count = cv2.countNonZero(thresh)
    
    # Classify the noise based on the number of white pixels
    if count > 0 and count < 1000:
        noise = 'Salt and Pepper'
    elif count >= 1000 and count < 5000:
        noise = 'Gaussian'
    elif count >= 5000:
        noise = 'Speckle'
    else:
        noise = 'No Noise'
    
    # Add the noise information to the list
    noises.append(noise)

# Convert the histograms and noise information to arrays
histograms = np.array(histograms)
noises = np.array(noises)

# Plot the histograms and noise information
for i in range(1):
    plt.subplot(2, 1, 1)
    plt.plot(histograms[:, i])
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency (pixels)')
    plt.title('Histogram')
    
    plt.subplot(2, 1, 2)
    plt.plot(noises)
    plt.xlabel('Frame Number')
    plt.ylabel('Noise Type')
    plt.title('Noise Information')
    
plt.tight_layout()
plt.show()
