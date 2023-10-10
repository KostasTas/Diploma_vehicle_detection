from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2
from vehicle_detector import VehicleDetector


model = load_model('ppico_2023_8_7_12_30_16.h5')

def predict(image_new):
    imagefile = image_new
    print("imagefile",imagefile)
    image_path = "static/imagesForUpload/" + imagefile.filename 
    imagefile.save(image_path) 

    # image = load_img(image_path, target_size=(64, 64)) 
    # image_array = img_to_array(image)
    # image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # confidence = model.predict(image_array)[0]
    
    # Check if the confidence is above the threshold for a vehicle
           # Load the original image with OpenCV for drawing bounding boxes
    original_image = cv2.imread(image_path)
    vd = VehicleDetector()

    vehicles_folder_count = 0
    vehicle_boxes = vd.detect_vehicles(original_image)
    vehicle_count = len(vehicle_boxes)


    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (25, 0, 180), 3)

        cv2.putText(original_image, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    cv2.imshow("Cars", original_image)
    cv2.waitKey(0)
