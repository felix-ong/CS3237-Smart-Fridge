from datetime import *
from dateutil.relativedelta import *
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
from numpy.random import randint
from random import random
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import *
from dateutil.relativedelta import *

from datetime import *
from dateutil.relativedelta import *
from scipy.interpolate import make_interp_spline

import io
import base64

import warnings
warnings.filterwarnings("ignore")

# -- Some helpers
    
def round_down(m,n):
    return m // n * n

def pos_and_rint(x):
    x = np.rint(x)
    return pos(x)

def pos(x):
    x[x<0] = 0
    return x

# rounds num_ago up to n, where n days ago was a day day_offset from Sunday
# returns first Sunday date and n
# day_offset = 0 for Sunday
def floorDayNumDaysAgo(now, num_ago, day_offset):
    ago = (now-timedelta(days=num_ago))
    previous = ago + relativedelta(weekday=SU(-1 + day_offset))
    diff = now - previous
    
    return previous, diff.days

def interp(x,y,num_pts):
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(min(x), max(x), num_pts)
    Y_ = X_Y_Spline(X_)
    return pos(X_),pos(Y_)

def GEN_DATA(days_ago, peak_consume, min_consume=0, weekend_peak=True, noise_sd=1, testing=False):
    
    # --- GENERATE DATA --- #
    '''
    Generate data that mimics a noisy periodic consumption that maxes at around peak_consume.
    
    days_ago is rounded so that days_ago from today lands on a Sunday.
    This is to align the peak consumption of our data on a consistent day of the week.
    Our assumption is that food consumption peaks on weekends and tapers off during mid-week.
    '''
    
    assert(days_ago > 3) # otherwise error on SARIMA

    mu, sig = 0, noise_sd # mean and standard deviation of noise
    noise = np.random.normal(mu, sig, days_ago)

    x = np.arange(0, days_ago)
    
    v_factor = (peak_consume - min_consume) / 2
    v_shift = (peak_consume + min_consume) / 2
    h_factor = ((2*np.pi) / 7) # get period len 7 (week)
    h_shift = (1/2)*np.pi if weekend_peak else (3/2)*np.pi # shift sundays to peak or trough
    y1 = np.sin(x*h_factor + h_shift) * v_factor + v_shift
    y = y1 + noise

    # discrete, positive values only
    y = pos_and_rint(y)
    
    return x,y

def SARIMA_PREDICT(y, orders, num_predict):

    # --- PREDICTION --- #

    ORDER, SEASONAL_ORDER = orders
    model = SARIMAX(y, order=ORDER, seasonal_order=SEASONAL_ORDER)
    model_fit = model.fit(disp=0)

    PREDICT_DAYS = num_predict

    yhat = model_fit.predict(len(y), len(y) + PREDICT_DAYS - 1)
    yhat = pos_and_rint(yhat)
        
    return yhat
    
# pass in optional date which is where data will start from
def DISPLAY_DATA(now, y, yhat, item_name):

    # --- PLOTTING --- #
    
    # amount of original data to display same as number of predicted
    # don't go below 14 and don't exceed original data length
    PREV_DAYS = min(max(len(yhat), 14), len(y))
    
    XTICK_INTERVAL = 7 # x-axis interval
    
    x1 = list(range(len(y)))
    x2 = list(range(len(y), len(y) + len(yhat)))
    
    plt.rcParams['figure.figsize'] = [15, 8]
    
    plot_x = x1[len(x1) - PREV_DAYS:] # truncate original data
    plot_y = y[len(y) - PREV_DAYS:]
    
    _, ax = plt.subplots()
        
    # interpolate to smooth
    interp_threshold = 45
    interp_factor = 4
    x1_,y_ = interp(plot_x,plot_y,len(plot_x)*interp_factor) if len(plot_x) > interp_threshold else (plot_x,plot_y)
    x2_,yhat_ = interp(x2,yhat,len(x2)*interp_factor) if len(x2) > interp_threshold else (x2,yhat)
    
    plt.plot(x1_, y_)
    plt.plot(x2_, yhat_)

    # ceil to sunday should be first x tick
    _, first_sunday_days_ago = floorDayNumDaysAgo(now, PREV_DAYS - 6, 0)
    x_tick = x1[len(x1) - first_sunday_days_ago]
    
    xTicks = np.arange(x_tick, x2[-1], XTICK_INTERVAL)
    yTicks = np.arange(min(y), np.ceil(max(max(y_), max(yhat_)) + 1))
        
    verticalTicks = xTicks
    curr_day = x1[-1] + 1
    xTicks = list(xTicks) + [curr_day] # add tick for today
    
    labels = [(now - timedelta(days=int(curr_day - d))).strftime("%a, %d/%m/%y") for d in xTicks]
    ax.set_xticks(xTicks, labels=labels)
        
    plt.title(f"{item_name.capitalize()} Prediction")
    plt.yticks(yTicks)
    plt.xticks(xTicks, rotation=70) # week interval x-axis
    plt.vlines(verticalTicks, min(min(y_), min(yhat_)), max(max(y_), max(yhat_)) + 1, "red", "dotted")
    plt.xlabel('Day')
    plt.ylabel(f'Predicted item consumption of {item_name}')
    plt.legend(['Original data (truncated)', 'Predicted', f"{XTICK_INTERVAL} day interval"])

    # save plot to jpg
    my_stringIObytes = io.BytesIO()
    plt.savefig('prediction.jpg')
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    return my_base64_jpgData

'''
FIRESTORE UTILS
'''

def delete_collection(coll_ref, batch_size):
    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        print(f'Deleting doc {doc.id} => {doc.get().to_dict()}')
        doc.delete()
        deleted = deleted + 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)

'''
each doc in consumption:
{
    'apple': int,
    'banana': int,
    'orange': int,
}
'''

def format_date(d):
    return d.strftime("%Y%m%d")

def backfill_consumption(n_data, db_ref):
    # orangeS
    x1, y1 = GEN_DATA(days_ago=n_data, noise_sd=1, weekend_peak=True, peak_consume=3.5, min_consume=1)
    # BANANAS
    _, y2 = GEN_DATA(days_ago=n_data, noise_sd=0.6, weekend_peak=True, peak_consume=2, min_consume=1)
    # APPLES
    _, y3 = GEN_DATA(days_ago=n_data, noise_sd=0.3, weekend_peak=False, peak_consume=1, min_consume=1)

    now = date.today()

    print(f"backfilling consumption from date: {now - timedelta(days=n_data)}")

    for i in range(len(x1)):
        timestamp = (format_date(now - timedelta(days=(n_data - i))))
        payload = {
            'orange': y1[i],
            'banana': y2[i],
            'apple': y3[i],
            'timestamp': timestamp,
        }
        doc_ref = db_ref.collection('consumption').document(timestamp)
        doc_ref.set(payload)

'''
Right before each prediction, add / update consumption in firestore.
Assume day T-1 just finished (we are currently in day T). 
Calculate the consumption: count(day T-2) - count(day T-1).

Again, this is just for DEMO purposes.

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
            'banana': doc2['banana'] - doc1['banana'] if doc2['banana'] - doc1['banana'] > 0 else 0,
            'apple': doc2['apple'] - doc1['apple'] if doc2['apple'] - doc1['apple'] > 0 else 0,
            'orange': doc2['orange'] - doc1['orange'] if doc2['orange'] - doc1['orange'] > 0 else 0,
            'timestamp': yesterday,
        })

    elif not doc1.exists and doc2.exists:
        # shouldn't happen
        pass
    else:
        pass # don't do anything, this will just use the 'fake' backfilled data