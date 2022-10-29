import paho.mqtt.client as mqtt
import cv2
import json
import os
import time
import pickle
from collections import defaultdict

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)



def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("fridge/photoresistor")

def on_message(client, userdata, msg):
    if int(msg.payload) > 500: # take photos every ?s until door is closed
        print("Fridge door is open!")
        time.sleep(2)
        # Take picture using camera, then run YOLO model and save counts to database
        result, image = cam.read()
        print("Photo taken!")
        image = json.dumps(image.tolist())
        client.publish("fridge/photo", image)
        time.sleep(50)
        # time.sleep(5)
        # if result:
            # TODO: figure out a way to save only the counts of that last frame before door close
            # Run command line
            # os.system(f"python detect.py --weights yolov7-e6e.pt --conf 0.5 --img-size 1280  --source {time1}.jpg --no-trace")
            # dict1 = defaultdict(lambda x: 0)
            # time.sleep(10)
            # with open("counts.txt", "r") as f:
            #     lines = f.read().splitlines()
            #     for line in lines:
            #         print(line.split(" "))
            #         name, count = line.rsplit(" ", 1)
            #         dict1[name] = int(count)
            #
            # # TODO: get this value from firebase/predicted
            # THRESHOLD = 2
            #
            # sensor_val = ""
            # firestore_payload = {}
            # for item in FRIDGE_ITEMS:
            #     if item == "banana":
            #         count = dict1[item] if item in dict1 else 0
            #         firestore_payload[item] = count
            #         if not count:  # count == 0
            #             sensor_val = "r"
            #         elif count < THRESHOLD:
            #             sensor_val = "y"
            #         else:
            #             sensor_val = "g"
            #         print(sensor_val)
            #
            # curr_time = time.time()
            # firestore_payload['timestamp'] = curr_time
            #
            # # TODO: make doc title day/month/year so that new updates coalesce into same doc
            # # that way we can use prev doc to use for count
            # # when we generate fake data, generate to yesterdays date
            # # also when there is no data for that day, copy over previous count for that day
            #
            # # send to firestore
            # doc_ref = db.collection(str('stocks')).document(str(curr_time))
            # doc_ref.set(firestore_payload)
            #
            # # pub to stock thread
            # client.publish("fridge/stock", sensor_val)
    else:
        print("Fridge door is closed.")
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting")
client.connect("test.mosquitto.org", 1883, 60)
client.loop_forever()

