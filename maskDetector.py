import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from imutils.video import VideoStream 
import serial.tools.list_ports as st
import numpy as np
import argparse as ap
import imutils
import serial
import time
import cv2
import os
import sys

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    sys.exit("Invalid device or cannot modify virtual devices once initialized.")

portNum = "NO-COM-FOUND"
for port in list(st.comports()):
	print("PORT:",port)
	if port[2].startswith('USB VID:PID=2341:8036'):
		portNum=port[0]
if portNum == "NO-COM-FOUND":
	sys.exit("Check if Arduino is connected.")
Arduino = serial.Serial(portNum, 9600)
time.sleep(1)

def ArduinoCall(_doorLocked):
	if _doorLocked == True:
		Arduino.write(b'1')
	else:
		Arduino.write(b'0')


def frameMaskHandler(frame, faceNet, maskNet):
	h, w = frame.shape[:2]
	faceNet.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)))
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			startX, startY, endX, endY = box.astype("int")
			startX, startY = (max(0, startX), max(0, startY))
			endX, endY = (min(w - 1, endX), min(h - 1, endY))
			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype = "float32")
		preds = maskNet.predict(faces, batch_size = 32)
	return locs, preds

cwd = os.getcwd()
CL_Args = ap.ArgumentParser()
CL_Args.add_argument("-f", "--face", type = str, default = cwd+"\\face_detector", help = "path to model directory")
CL_Args.add_argument("-m", "--model", type = str, default = cwd+"\mask.model", help = "path to trained model")
CL_Args.add_argument("-c", "--confidence", type = float, default = 0.8, help = "minimum probability to filter weak detections")
args = vars(CL_Args.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])
stream = VideoStream(src = 0).start()
time.sleep(2.0)

_doorLockedPrior = False
while True:
	_doorLockedINNER = False
	frame = imutils.resize(stream.read(), width = 1000)
	locs, preds = frameMaskHandler(frame, faceNet, maskNet)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		if mask > withoutMask:
			label = "Mask"
			_doorLockedINNER = True
		else:
			label = "No Mask"
			_doorLockedINNER = False
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	if _doorLockedPrior != _doorLockedINNER:
		print("Mask Value Changed:", _doorLockedINNER)
		ArduinoCall(_doorLockedINNER)
		_doorLockedPrior = _doorLockedINNER
	cv2.imshow("Mask Check", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
cv2.destroyAllWindows()
stream.stop()
Arduino.close()