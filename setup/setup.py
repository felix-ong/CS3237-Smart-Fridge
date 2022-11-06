from helpers import *

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

from datetime import *
from dateutil.relativedelta import *

from stock_predict import *
import os

import time
from threading import Thread

cred = credentials.Certificate('cs3237-fridge-firebase-adminsdk-d14fo-88295eb35b.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

ITEMS = ['banana', 'apple', 'orange']

# TODO: make sure that deleting on collection that doesnt exist doesnt err.

'''
Backfill consumptions for each item
Delete the collection before each time.
'''
delete_collection(coll_ref=db.collection('consumption'), batch_size=60)
backfill_consumption(n_data=60, db_ref=db) # adds orange, apple, banana consumption fake data

'''
JUST FOR DEMO - fill previous 2 day's counts on setup
when prediction runs it'll calc consumption for 'yesterday' and update firebase
'''
delete_collection(coll_ref=db.collection('stocks'), batch_size=60)
now = date.today()
yesterday, two_days_ago = format_date(now - timedelta(days=1)), format_date(now - timedelta(days=2))
doc_ref = db.collection('stocks').document(yesterday)
doc_ref.set({
    'apple': 2,
    'banana': 1,
    'orange': 1,
    'timestamp': yesterday,
})
doc_ref = db.collection('stocks').document(two_days_ago)
doc_ref.set({
    'apple': 0,
    'banana': 1,
    'orange': 1,
    'timestamp': two_days_ago,
})

'''
Add default config
'''
config_ref = db.collection('config').document('config')
config_ref.set({
    'prediction_window': 7,
    'historical_window': 60,
})

'''
Run prediction with values in firebase/config or defaulted vals = (60, 7).
Note: Less data might exist than specified by historical_window.

TODO: predict should listen for a change in values to config (when app updates slider). then rerun ..
'''

# runs predictions on apples, bananas, oranges
# also calculates consumption right before
calc_yesterdays_consumption(db, now)
stock_predict(db)

# ALSO TODO: add the photo in imgur link to send to firebase just
# put a listener on the app to display the picture

# TODO: limits for prediction params on app should be min 14 for historical
'''
Start 2 threads:
    one should loop and run stock_predict at midnight
    one should run and listen for config changes to rerun stock_predict
'''

def daily_predict():
    print('starting daily predict')
    currdate = date.today().strftime("%d")
    while True:
        time.sleep(3600) # every hour
        if date.today().strftime("%d") != currdate:
            # run daily prediction
            stock_predict(db)
    
# TODO: modify so we are only running on change i.e. add listener
# https://firebase.google.com/docs/firestore/query-data/listen
# caveat: callback triggers before data is written, 
# need to add a delay because changes in metadata not supported in python client yet
def predict_on_config_change():
    print('detecting config changes')
    curr_p, curr_h = 7, 60 # begin to check against default
    while True:
        time.sleep(3)
        config = db.collection('config').document('config').get().to_dict()
        h, p = config['historical_window'], config['prediction_window']
        if h != curr_h or p != curr_p:
            stock_predict(db)
            # update local vals
            curr_h, curr_p = h, p

Thread(target=predict_on_config_change).start()
Thread(target=daily_predict).start()

# Then start running detect.py in new process
os.system('python yolov7/detect.py') # TODO: change path
