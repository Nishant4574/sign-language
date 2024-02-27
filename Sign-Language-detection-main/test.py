import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

class HandGestureRecognition:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.imgSize = 224  # Adjusted image size to match the input shape of the model
        self.labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

    def get_frame(self):
        ret, img = self.cap.read()
        if not ret:
            return False, None
        
        hands, img = self.detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            # Check if imgCrop is empty before resizing
            if not imgCrop.size == 0:
                # Resize the cropped image directly before classification
                imgResize = cv2.resize(imgCrop, (self.imgSize, self.imgSize))

                # Classify the resized image
                prediction, index = self.classifier.getPrediction(imgResize, draw=False)

                cv2.rectangle(img, (x - self.offset, y - self.offset - 70), (x - self.offset + 400, y - self.offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, self.labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(img, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (0, 255, 0), 4)

        return True, img

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
