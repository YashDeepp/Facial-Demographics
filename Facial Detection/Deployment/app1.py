from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image
app = Flask(__name__)

model_age=load_model("D://DIP//Facial Demographics//Facial Detection//Deployment//models//age_model_50epochs.h5")
model_emotion=load_model("D://DIP//Facial Demographics//Facial Detection//Deployment//models//emotion_epoch75.h5")
model_gender=load_model("D://DIP//Facial Demographics//Facial Detection//Deployment//models//gender_model_50epochs.h5")

model_age.make_predict_function()
model_gender.make_predict_function()
model_emotion.make_predict_function()

gender_labels = {0 : 'Male', 1 : 'Female'}
emotion_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

def predict_label(img_path):
    roi_color = np.array(tf.keras.utils.load_img(img_path, target_size=(200,200)))
    roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
    age_predict = model_age.predict(np.array(roi_color).reshape(-1,200,200,3))
    age = round(age_predict[0,0])

    gender_predict = model_gender.predict(np.array(roi_color).reshape(-1,200,200,3))
    gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
    gender=gender_labels[gender_predict[0]]

    roi_color = np.array(tf.keras.utils.load_img(img_path, target_size=(200,200)))
    roi_gray=cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    roi=roi_gray.astype('float')/255.0
    roi=tf.keras.preprocessing.image.img_to_array(roi)
    roi=np.expand_dims(roi,axis=0)
    preds=model_emotion.predict(roi)[0]
    emotion=emotion_labels[preds.argmax()]

    return age,gender,emotion

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        img= request.files['my_image']
        img_path = "uploads//" + img.filename
        img.save(img_path)
        age,gender,emotion= predict_label(img_path)
        return render_template("index.html", prediction = [age,gender,emotion], img_path = img_path)
    return None

if __name__ == '__main__':
    app.run(debug=True)
