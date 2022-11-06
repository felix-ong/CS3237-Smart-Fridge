from helpers import *
from datetime import *

MODEL_ORDERS = ((1,0,0), (1, 1, 1, 7))

'''
When prediction runs, will send data to firebase + graph.
App should hook into changes to prediction and update.
'''

# predicts and sends graphs + data to firestore
def stock_predict(db):
    now = date.today()

    # when we run predictions at midnight, want to calculate yesterdays consumption before
    calc_yesterdays_consumption(db, now)

    # -- MAKE PREDICTIONS -- #

    historical_window, prediction_window = fetch_params(db)

    query_from = format_date(now - timedelta(days=historical_window))

    docs = db.collection(u'consumption').where(u'timestamp', u'>=', query_from).stream()

    apple_data, banana_data, egg_data = [], [], []
    for doc in docs:
        counts = doc.to_dict()
        apple_data.append(counts['apple'])
        banana_data.append(counts['banana'])
        egg_data.append(counts['egg'])

    assert(len(apple_data) > 7, 'Not enough consumption data!')

    predict_for_item(db, apple_data, prediction_window, 'apple', now)
    predict_for_item(db, banana_data, prediction_window, 'banana', now)
    predict_for_item(db, egg_data, prediction_window, 'egg', now)

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

'''
Right before each prediction, add / update consumption in firestore.
Assume day T-1 just finished (we are currently in day T). 
Calculate the consumption: count(day T-2) - count(day T-1).

If there are no counts for day T-2 or day T-1, just use fake consumption data (only true its truly first time setup).
'''
def calc_yesterdays_consumption(db, now):
    yesterday, two_days_ago = format_date(now - timedelta(days=1)), format_date(now - timedelta(days=2))

    # fetch stocks for day T-1 and T-2
    doc1 = db.collection('stocks').document(yesterday).get()
    doc2 = db.collection('stocks').document(two_days_ago).get()
    if doc1.exists and doc2.exists:
        doc1 = doc1.to_dict()
        doc2 = doc2.to_dict()

        doc_ref = db.collection('consumption').document(yesterday)

        # calculate and update consumptions in firestore
        doc_ref.set({
            'banana': doc2['banana'] - doc1['banana'],
            'apple': doc2['apple'] - doc1['apple'],
            'egg': doc2['egg'] - doc1['egg'],
            'timestamp': yesterday,
        })

    elif not doc1.exists and doc2.exists:
        # didn't open frige yesterday, consumption was 0.

        doc_ref = db.collection('consumption').document(yesterday)
        doc_ref.set({
            'banana': 0,
            'apple': 0,
            'egg': 0,
            'timestamp': yesterday,
        })

    else:
        pass # don't do anything, this will just use the 'fake' backfilled data
