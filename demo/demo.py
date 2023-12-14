import cv2
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import os
from codecarbon import track_emissions
import labels  # Import the labels module with your classes

@track_emissions()
def object_detection():

    # Load model
    model = torch.load('model.pth', map_location=torch.device('cpu'))
    model.eval()

    # Set up the transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open a connection to the camera (assuming camera index 0)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert OpenCV BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Preprocess the image
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)

        # Convert the output to probabilities and get the predicted class index
        probabilities = F.softmax(output[0], dim=0)
        class_index = torch.argmax(probabilities).item()

        # Get the class label
        class_label = labels.classes[class_index]

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        print(f'Detected: \033[1m {class_label} \033[0m')

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


object_detection()

