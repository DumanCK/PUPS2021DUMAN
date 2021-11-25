import cv2
import imutils

net = cv2.dnn.readNetFromTensorflow("modelpb.pb")
frame = cv2.imread('test/5.jpg')

img = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224), (104, 117, 123))

net.setInput(img)
detections = net.forward()
print(detections)
