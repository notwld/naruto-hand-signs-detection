import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Define the labels for the classes
classes = ['bird', 'boar', 'dog', 'dragon', 'hare', 'horse', 'monkey', 'ox', 'ram', 'rat', 'snake', 'tiger', 'zero']

# Define the function for detecting hand signs in real-time
def detect_hand_sign():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Set the size of the camera output
    cap.set(3, 640) # set width
    cap.set(4, 480) # set height

    while True:
        # Capture the frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale frame to RGB
        frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Apply a Gaussian blur to the frame
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive thresholding to the frame
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If any contours are found
        if len(contours) > 0:
            # Find the largest contour
            contour = max(contours, key=cv2.contourArea)

            # If the contour is big enough
            if cv2.contourArea(contour) > 5000:
                # Create a bounding rectangle around the contour
                (x, y, w, h) = cv2.boundingRect(contour)

                # Extract the ROI (Region of Interest) from the frame
                roi = frame_rgb[y:y+h, x:x+w]

                # Resize the ROI to the size of the input to the model
                roi = cv2.resize(roi, (128, 128))

                # Reshape the ROI to the shape expected by the model
                roi = roi.reshape(1, 128, 128, 3)

                # Normalize the ROI
                roi = roi / 255.0

                # Use the pre-trained model to predict the class of the ROI
                prediction = model.predict(roi)

                # Get the predicted class label
                label = classes[np.argmax(prediction)]

                # Draw the predicted label on the top left corner of the frame
                cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                

        # Display the frame
        cv2.imshow('Hand Sign Detection', frame)

        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()

    cv2.destroyAllWindows()

# Call the function to detect hand signs in real-time
detect_hand_sign()
