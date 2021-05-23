import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from flask import Flask,Response
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import cv2

model = None
def getModel(modelpath):
    global model
    model = load_model(modelpath)
    print(f"[+]loaded model {model}")

app=Flask(__name__,static_folder="build",static_url_path='/')

prototxt = "face_detector/deploy.prototxt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
modelpath = "model/keras_model.h5"
print("[+]loading the model")
getModel(modelpath)


faceNet = cv2.dnn.readNet(prototxt,weights)
def getMask(frame):
    #takin the dimension of the image(h,w,r)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detection = faceNet.forward()

    faces = []
    locs = []
    preds = []
    conf = 0.4
    for i in range(0,detection.shape[2]):
        #extracting the confidence
        confidence = detection[0,0,i,2]
        if confidence > conf:
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            #making in coordination System
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #taking out the face from the frame
            face = frame[startY:endY,startX:endX]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY,endX,endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = model.predict(faces, batch_size=32)
        else:
            pass
        return (locs,preds)


camera = cv2.VideoCapture(0)
def get_frame():
    while True:
        success,frame = camera.read()
        if not success:
            break
        else:
            (locs,preds) = getMask(frame)
            for (box,pred) in zip(locs,preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                ret,buffer =cv2.imencode('.jpg',frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/api/video_feed")
def video_feed():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def get_current_time():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run()