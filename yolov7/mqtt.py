import paho.mqtt.client as mqtt
from flask import Flask, request
import cv2
import os
import time

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

app = Flask(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("fridge/photoresistor")

def on_message(client, userdata, msg):
    if int(msg.payload) > 500:
        print("Fridge door is open!")
        time.sleep(5)
        # Take picture using camera, then run YOLO model and save counts to database
        result, image = cam.read()
        print("Photo taken!")
        save_path = os.path.join("test.jpg")
        cv2.imwrite(save_path, image)
        if result:
            # Run command line
            os.system("python detect.py --weights yolov7-e6e.pt --conf 0.5 --img-size 1280  --source test.jpg --no-trace >> out.txt")
            # TODO
            with open("counts.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    name, count = line.split(" ")
                    print(f"{count} {name}s")
            # Send time, item name, count to database, publish to fridge/stock
    else:
        print("Fridge door is closed.")
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting")
client.connect("test.mosquitto.org", 1883, 60)
client.loop_forever()

