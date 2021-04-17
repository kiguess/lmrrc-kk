# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from os import path
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import gc

import pickle
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))


# %%
# Import data
with open(path.join(BASE_DIR, "data/model_build_inputs/route_data.json"), "r") as rou:
    sample_r = json.load(rou)

with open(path.join(BASE_DIR, "data/model_build_inputs/travel_times.json"), "r") as trav:
    sample_t = json.load(trav)

with open(path.join(BASE_DIR, "data/model_build_inputs/actual_sequences.json"), "r") as act:
    act_seq = json.load(act)


# %%
# Data Processing
def create_seq(seq: dict):
    out = {}
    for route in seq:
        sorted_list  = []
        act          = seq[route]['actual']
        sorted_dict  = dict(sorted(act.items(), key=lambda item: item[1]))
        for key in sorted_dict:
            sorted_list.append(key)
        out[route] = sorted_list
    return out

def create_comb(rou: dict, seq: dict, trav: dict):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    temp        = open("temp", "w")
    header      = ['RouteID', 'station_code', 'date', 'departure_time', 'from', 'to']
    header.extend(['origin_lat', 'origin_long', 'dest_lat', 'dest_long', 'delta_lat', 'delta_long'])
    header.extend(['time_taken', 'score'])
    header      = ','.join(header)
    temp.write(header+'\n')

    for route in rou:
        stat_code = rou[route]['station_code']
        date      = rou[route]['date_YYYY_MM_DD']
        dep_time  = rou[route]['departure_time_utc']
        dep_timed = datetime.strptime(dep_time, '%H:%M:%S')
        for i in range(0, len(seq[route])-1):
            stop0   = seq[route][i]
            stop1   = seq[route][i+1]
            st0_lat = rou[route]['stops'][stop0]['lat']
            st0_lng = rou[route]['stops'][stop0]['lng']
            st1_lat = rou[route]['stops'][stop1]['lat']
            st1_lng = rou[route]['stops'][stop1]['lng']
            dlat    = st1_lat - st0_lat
            dlong   = st1_lng - st0_lng
            time    = trav[route][stop0][stop1]
            scorest = rou[route]['route_score'] 
            score   = 9 if scorest=='High' else 5 if scorest=='Medium' else 1

            li      = [route, stat_code, date, dep_time, stop0, stop1]
            li.extend([str(st0_lat), str(st0_lng), str(st1_lat), str(st1_lng), str(dlat), str(dlong)])
            li.extend([str(time), str(score)])

            out     = ','.join(li)
            temp.write(out + '\n')
            
            dep_timed = dep_timed + relativedelta(seconds=time)
            dep_time  = datetime.strftime(dep_timed, '%H:%M:%S')
            del(li)
            del(out)
    temp.close()
    return pd.read_csv('temp')
        


seq        = create_seq(act_seq)
route_data = create_comb(sample_r, seq, sample_t)

del(sample_r)
del(sample_t)
del(act_seq)
del(seq)
gc.collect()

route_data.head()


# %% [markdown]
# # Modeling

# %%
X = sample_df.drop(["trip_duration", "id", "vendor_id", "pickup_datetime", "dropoff_datetime"], axis=1)
y = sample_df["trip_duration"]


# %%
#Split the data into training, test, and valdiation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2019)


# %%
#Define evaluation metric
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


# %%
#XGBoost parameters 
params = {
    'booster':            'gbtree',
    'objective':          'reg:linear',
    'learning_rate':      0.05,
    'max_depth':          14,
    'subsample':          0.9,
    'colsample_bytree':   0.7,
    'colsample_bylevel':  0.7,
    'silent':             1,
    'feval':              'rmsle'
}


# %%
nrounds = 2000


# %%
#Define train and validation sets
dtrain = xgb.DMatrix(X_train, np.log(y_train+1))
dval = xgb.DMatrix(X_val, np.log(y_val+1))

#this is for tracking the error
watchlist = [(dval, 'eval'), (dtrain, 'train')]


# %%
#Train model
gbm = xgb.train(params,
                dtrain,
                num_boost_round = nrounds,
                evals = watchlist,
                verbose_eval = True
                )


# %%
#Test predictions
pred = np.exp(gbm.predict(xgb.DMatrix(X_test))) - 1


# %%
#Use mean absolute error to get a basic estimate of the error
mae = (abs(pred - y_test)).mean()
mae


# %%
#Take a look at feature importance
feature_scores = gbm.get_fscore()
feature_scores


# %%
#This is not very telling, so let's scale the features
summ = 0
for key in feature_scores:
    summ = summ + feature_scores[key]

for key in feature_scores:
    feature_scores[key] = feature_scores[key] / summ

feature_scores

# %% [markdown]
# # Save the model

# %%
filename = "xgb_model.sav"
pickle.dump(gbm, open(filename, 'wb'))


