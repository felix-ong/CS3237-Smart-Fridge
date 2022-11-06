import paho.mqtt.client as mqtt
import cv2
import json
import time
import numpy as np

from threading import Thread
from threading import Lock

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
photo_lock = Lock()

def take_photos():
    # discard job if another is running photos (don't add to waiting queue)
    # we don't want to act on old fridge values
    ret = photo_lock.acquire(blocking=False)
    if not ret:
        return

    prevImg = None
    while True:
        # take photos every x seconds until door is closed
        time.sleep(0.5)
        # Take picture using camera, then run YOLO model and save counts to database
        res, image = cam.read()

        if not res: continue # retry until camera is working

        print(f"Photo taken!\nSize: {image.shape}")

        # if image is dark, send prevImg to process (if prevImg exists and is not also dark)
        if is_dark_image(image) and isinstance(prevImg, np.ndarray) and not is_dark_image(prevImg):
            print(f"Detected a dark image. Sending last usable picture..")

            client.publish("fridge/photo", json.dumps(prevImg.tolist()))
            break

        prevImg = image

    photo_lock.release()

def is_dark_image(img):
    v = np.average(img)
    print(f"Dark value: {v}")
    return v < 40

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("fridge/photoresistor")

prev = -1

def on_message(client, userdata, msg):
    global prev
    door_open = 500
    if int(msg.payload) > door_open:
        if int(prev) <= door_open:
            print("Fridge door is open!")
            # spawn a new thread to take care of the camera photos
            # so we can continue to receive new messages
            Thread(target=take_photos).start()
    else:
        if int(prev) == -1 or int(prev) > door_open:
            print("Fridge door is closed.")

    prev = msg.payload
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting")
client.connect("192.168.199.37", 1883, 60)
client.loop_forever()
