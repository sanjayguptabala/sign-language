import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('C:/Users/sanja/PycharmProjects/pythonProject/model.h5')


# Define the labels for the ISL alphabets and numbers
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
          10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
          19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '1', 27: '2',
          28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9'}

# Capture video from the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break;
        
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocess the ROI for prediction
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = roi.reshape(1, 64, 64, 1)
    roi = roi / 255.0

    # Make a prediction
    prediction = model.predict(roi)
    max_index = int(np.argmax(prediction))
    predicted_label = labels[max_index]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('ISL Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
