from helpers import *

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

from datetime import *
from dateutil.relativedelta import *

from stock_predict import *

cred = credentials.Certificate('../yolov7/cs3237-fridge-firebase-adminsdk-d14fo-88295eb35b.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

ITEMS = ['banana', 'apple', 'egg']

# TODO: make sure that deleting on collection that doesnt exist doesnt err.

'''
Backfill consumptions for each item
Delete the collection before each time.
'''
delete_collection(coll_ref=db.collection('consumption'), batch_size=60)
backfill_consumption(n_data=120, db_ref=db) # adds egg, apple, banana consumption fake data

'''
JUST FOR DEMO - fill previous 2 day's counts on setup
when prediction runs it'll calc consumption for 'yesterday' and update firebase
'''
delete_collection(coll_ref=db.collection('stocks'), batch_size=60)
now = date.today()
yesterday, two_days_ago = format_date(now - timedelta(days=1)), format_date(now - timedelta(days=2))
doc_ref = db.collection('stocks').document(yesterday)
doc_ref.set({
    'apple': 3,
    'banana': 4,
    'egg': 3,
    'timestamp': yesterday,
})
doc_ref = db.collection('stocks').document(two_days_ago)
doc_ref.set({
    'apple': 5,
    'banana': 5,
    'egg': 5,
    'timestamp': two_days_ago,
})

'''
Delete config.
'''
delete_collection(coll_ref=db.collection('config'), batch_size=60)

'''
Run prediction with values in firebase/config or defaulted vals = (60, 7).
Note: Less data might exist than specified by historical_window.

TODO: predict should listen for a change in values to config (when app updates slider). then rerun ..
'''

# runs predictions on apples, bananas, eggs
# also calculates consumption right before
stock_predict(db)

# ALSO TODO: add the photo in imgur link to send to firebase just
# put a listener on the app to display the picture

# TODO: limits for prediction params on app should be min 14 for historical
'''
Start running detect.py ??

Also start 2 threads:
    one should loop and run stock_predict at midnight
    one should run and listen for config changes to rerun stock_predict
'''