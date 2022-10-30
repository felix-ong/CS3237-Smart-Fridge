import paho.mqtt.client as mqtt
import cv2
import json
import time
import numpy as np


cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

prevImg = None

def is_dark_image(img):
    v = np.average(img)
    print(f"Dark value: {v}")
    return v < 40

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("fridge/photoresistor")

def on_message(client, userdata, msg):
    if int(msg.payload) > 500: # take photos every 2s until door is closed
        while True:
            time.sleep(2)
            # Take picture using camera, then run YOLO model and save counts to database
            result, image = cam.read()

            print(f"Photo taken!\nSize: {image.shape}")

            # if image is dark, send prevImg to process (if prevImg exists and is not also dark)
            if is_dark_image(image) and isinstance(prevImg, np.ndarray) and not is_dark_image(prevImg):
                print(f"Detected a dark image. Sending last usable picture..")

                client.publish("fridge/photo", json.dumps(prevImg.tolist()))
                break

            prevImg = image

    else:
        print("Fridge door is closed.")
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting")
client.connect("172.25.104.209", 1883, 60)
client.loop_forever()

