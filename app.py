import base64
import os
from flask import Flask, render_template, request, url_for
from predictor import predict
from flask_mail import Mail, Message
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2
from vehicle_detector import VehicleDetector


author = 'TEAM DELTA'

app = Flask(__name__, static_folder="static")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'kostas2113@gmail.com'
app.config['MAIL_PASSWORD'] = 'nmivkrxpnbpcsoii'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


model = load_model('ppico_2023_8_7_12_30_16.h5')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')


@app.route('/send_email', methods = ['GET', 'POST'])
def email():
    if request.method == 'POST':
        try:
            reciepent = request.form['email']
            body = request.form['message']
            msg = Message("Car Thesis", sender='noreply@demo.com', recipients = [reciepent] )
            msg.body = body
            mail.send(msg)
            return render_template('contact.html', success=True)
        except Exception as e:
            return render_template('contact.html', message=str(e), failure=True)
           
@app.route('/result', methods=['GET', 'POST'])
def result():
    imagefile = request.files['file'] 
    image_path = "static/imagesForUpload/" + imagefile.filename 
    imagefile.save(image_path) 

    image = load_img(image_path, target_size=(64, 64)) 
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    confidence = model.predict(image_array)[0]
    
    
    # Load the original image with OpenCV for drawing bounding boxes
    original_image = cv2.imread(image_path)
    test_image = original_image
    vd = VehicleDetector()

    vehicles_folder_count = 0
    vehicle_boxes = vd.detect_vehicles(original_image)
    vehicle_count = len(vehicle_boxes)


    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (25, 0, 180), 3)

        # cv2.putText(original_image, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    # cv2.imshow("Cars", original_image)
    # cv2.waitKey(0)
    image_content = cv2.imencode('.jpg', original_image)[1].tostring()
    # Create base64 encoding of the string encoded image
    encoded_image = base64.b64encode(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    # Check if the confidence is above the threshold for a vehicle
    if confidence >= 0.5:
        return render_template('result.html', image_name=to_send, detection=True, confidence_score=confidence)
    else:
        return render_template('result.html', image_name=to_send, detection=False, confidence_score=confidence)
