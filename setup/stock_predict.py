from helpers import *
from datetime import *

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

MODEL_ORDERS = ((1,0,0), (1, 1, 1, 7))

'''
When prediction runs, will send data to firebase + graph.
App should hook into changes to prediction and update.
'''

# predicts and sends graphs + data to firestore
def stock_predict(db):
    now = date.today()

    # -- MAKE PREDICTIONS -- #

    historical_window, prediction_window = fetch_params(db)

    query_from = format_date(now - timedelta(days=historical_window))
    curr_date = format_date(now)

    docs = db.collection(u'consumption').where(u'timestamp', u'>=', query_from).where(u'timestamp', u'<', curr_date).stream()

    apple_data, banana_data, orange_data = [], [], []
    for doc in docs:
        counts = doc.to_dict()
        apple_data.append(counts['apple'])
        banana_data.append(counts['banana'])
        orange_data.append(counts['orange'])

    predict_for_item(db, apple_data, prediction_window, 'apple', now)
    predict_for_item(db, banana_data, prediction_window, 'banana', now)
    predict_for_item(db, orange_data, prediction_window, 'orange', now)

def fetch_params(db):
    # fetch prediction parameters from firebase
    config = db.collection('config').document('config').get()
    if config.exists:
        conf = config.to_dict()
        return conf['historical_window'], conf['prediction_window']
    else:
        return 60, 7 # default, in case config doesn't exist

# predicts and saves to firebase for item
def predict_for_item(db, data, prediction_window, item, now):
    predicted = SARIMA_PREDICT(data, orders=MODEL_ORDERS, num_predict=prediction_window)

    graph_img = DISPLAY_DATA(now, data, predicted, item_name=item)

    doc_ref = db.collection('predict').document(item)
    doc_ref.set({
        'historical_data': list(data),
        'predicted_data': predicted.tolist(),
        'graph_img_base64': graph_img,
    })

if __name__ == "__main__":
    cred = credentials.Certificate('cs3237-fridge-firebase-adminsdk-d14fo-88295eb35b.json')
    app = firebase_admin.initialize_app(cred)
    db = firestore.client()

    stock_predict(db)
