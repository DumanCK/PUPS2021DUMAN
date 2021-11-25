from flask import Flask, render_template, Response
import cv2
#from imutils.video import VideoStream
#from imutils.video import FPS
import numpy as np
#import argparse
import imutils
#import time

COLORS = np.random.uniform(0, 255, size=1)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow("modelpb.pb")
print("[INFO] starting video stream...")

app = Flask(__name__)

camera = cv2.VideoCapture('https://media.lewatmana.com/cam/sotisresidence/331/videoclip20210917_084835.384.mp4')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    c = 0
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224), (123.68, 116.779, 103.939))
            
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            print(detections)
            kelas = np.argmax(detections, axis=1)
            print(kelas[0])
            if (c==0):
                if (kelas[0] == 1):
                    teks = "Tabrakan"
                    c = 90
                elif (kelas[0] == 2):
                    teks = "Tidak Tabrakan"    
                elif (kelas[0] == 0):
                    teks = "Netral"
            else :
                teks = "Tabrakan"
                c -= 1
            cv2.putText(frame, teks, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,0], 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
