import paho.mqtt.client as mqtt
import cv2
import os
import time
import datetime
import json
from collections import defaultdict

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

cred = credentials.Certificate('cs3237-fridge-firebase-adminsdk-d14fo-88295eb35b.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

doc_ref = db.collection(str('stocks')).document(str(time.time()))
doc_ref.set({
    'bottles': 0,
    'bananas': 0,
    'apples': 0,
    'timestamp': time.time(),
})



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
            dict = defaultdict(lambda x: 0)
            with open("counts.txt", "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    print(line.split(" "))
                    name, count = line.rsplit(" ", 1)
                    dict[name] = count

            # send to firebase
            doc_ref = db.collection(str('stocks')).document(str(time.time()))
            doc_ref.set({
                'bottles': 0,
                'bananas': 0,
                'apples': 0,
                'timestamp': time.time(),
            })

            THRESHOLD = 2  # TODO: get this value from firebase/predicted
            sensor_val = ""
            for item in ['bananas', 'apples', 'bottles']:
                count = dict[item] if item in dict else 0
                if not count:  # count == 0
                    sensor_val = "r"
                elif count < THRESHOLD:
                    sensor_val = "y"
                else:
                    sensor_val = "g"

            client.publish("fridge/stock", sensor_val)
    else:
        print("Fridge door is closed.")
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting")
client.connect("test.mosquitto.org", 1883, 60)
client.loop_forever()

