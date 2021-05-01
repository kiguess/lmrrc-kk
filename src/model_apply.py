# %%
from os import path
import json, time, gc, logging
import xgboost as xgb
from pandas import DataFrame
from numpy import exp

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
output_path  = str(path.join(BASE_DIR, "data/model_apply_outputs/"))
logfile  = path.join(output_path, 'apply.log')

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logfile, encoding='utf-8', level=logging.DEBUG, filemode='w')
logging.info('Started')


# %%
# Load model
logging.info('Loading model')
model_path = path.join(BASE_DIR, 'data/model_build_outputs/model.json')
model      = xgb.Booster()
model.load_model(model_path)


# %%
# Load new data to predict
logging.info('Loading data')
prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
with open(prediction_routes_path, newline='') as in_file:
    prediction_routes = json.load(in_file)

travel_times_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
with open(travel_times_path, 'r') as file:
    travel_times = json.load(file)
logging.info('Data loaded')


# %%
# Functions that might be useful
def sort_by_key(stops, sort_by):
    """
    Takes in the `prediction_routes[route_id]['stops']` dictionary
    Returns a dictionary of the stops with their sorted order always placing the depot first

    EG:

    Input:
    ```
    stops={
      "Depot": {
        "lat": 42.139891,
        "lng": -71.494346,
        "type": "depot",
        "zone_id": null
      },
      "StopID_001": {
        "lat": 43.139891,
        "lng": -71.494346,
        "type": "delivery",
        "zone_id": "A-2.2A"
      },
      "StopID_002": {
        "lat": 42.139891,
        "lng": -71.494346,
        "type": "delivery",
        "zone_id": "P-13.1B"
      }
    }

    print (sort_by_key(stops, 'lat'))
    ```

    Output:
    ```
    {
        "Depot":1,
        "StopID_001":3,
        "StopID_002":2
    }
    ```

    """
    # Serialize keys as id into each dictionary value and make the dict a list
    stops_list=[{**value, **{'id':key}} for key, value in stops.items()]

    # Sort the stops list by the key specified when calling the sort_by_key func
    ordered_stop_list=sorted(stops_list, key=lambda x: x[sort_by])

    # Keep only sorted list of ids
    ordered_stop_list_ids=[i['id'] for i in ordered_stop_list]

    # Serialize back to dictionary format with output order as the values
    return {i:ordered_stop_list_ids.index(i) for i in ordered_stop_list_ids}

def propose_all_routes(prediction_routes, sort_by):
    """
    Applies `sort_by_key` to each route's set of stops and returns them in a dictionary under `output[route_id]['proposed']`

    EG:

    Input:
    ```
    prediction_routes = {
      "RouteID_001": {
        ...
        "stops": {
          "Depot": {
            "lat": 42.139891,
            "lng": -71.494346,
            "type": "depot",
            "zone_id": null
          },
          ...
        }
      },
      ...
    }

    print(propose_all_routes(prediction_routes, 'lat'))
    ```

    Output:
    ```
    {
      "RouteID_001": {
        "proposed": {
          "Depot": 0,
          "StopID_001": 1,
          "StopID_002": 2
        }
      },
      ...
    }
    ```
    """
    return {key:{'proposed':sort_by_key(stops=value['stops'], sort_by=sort_by)} for key, value in prediction_routes.items()}

def get_station(route_data: dict) -> str:
    'Return the name of the station in a route.'
    for stops in route_data['stops']:
        if (route_data['stops'][stops]['type'] == 'Station'):
            return stops


# %%
# Travel times sorting
logging.info('Making a list for travel times')
def sorted_key(to_sort: dict) -> list:
    'Return list with keys already sorted'
    return sorted(to_sort, key=to_sort.get)


def sort_travel_times(travel_times: dict) -> dict:
    '''
    Takes the travel times and sort by shortest time for each stops to go to another stop.

    Return a dict containing sorted lists from each stops on each routes
    '''
    sorted_trav = {}
    for route in travel_times:
        sorted_trav[route] = {}
        for stops in travel_times[route]:
            sorted_trav[route][stops] = []
            sorted_trav[route][stops] = sorted_key(travel_times[route][stops])
            sorted_trav[route][stops].pop(0)
    
    return sorted_trav

travel_times_sorted = sort_travel_times(travel_times)
logging.info('Travel times list done')
gc.collect()


# %%
# Functions for proposal
def create_dmatrix(route_data: dict, travel: dict, origin: str, dest: str, time_passed: float=0) -> xgb.DMatrix:
    '''
    Return a DMatrix containing all data for training of specific part of a route
    Params:
    route_data  = route data of THIS route only
    travel      = Travel time of this route
    origin      = travel from which stop
    dest        = travel to which stop
    time_passed = time passed in this route (in seconds)
    '''
    import datetime
    from dateutil.relativedelta import relativedelta

    dep_time = datetime.datetime.strptime(route_data['departure_time_utc'], '%H:%M:%S') + relativedelta(seconds=time_passed)
    hour    = dep_time.hour + dep_time.minute/60
    st0_lat = route_data['stops'][origin]['lat']
    st0_lng = route_data['stops'][origin]['lng']
    st1_lat = route_data['stops'][dest]['lat']
    st1_lng = route_data['stops'][dest]['lng']
    dlat    = st1_lat - st0_lat
    dlong   = st1_lng - st0_lng
    time    = travel[origin][dest]

    column_name = ['hour', 'origin_lat', 'origin_long', 'dest_lat', 'dest_long', 'delta_lat', 'delta_long', 'time_taken']
    value       = [hour, st0_lat, st0_lng, st1_lat, st1_lng, dlat, dlong, time]
    df          = DataFrame(value, column_name).transpose()

    return xgb.DMatrix(df, enable_categorical=True)


def create_proposal(route_data: dict, travel: dict, travel_sort: dict) -> list:
    '''
    Create a list containing the proposed sequence for the route
    Params:
    route_data : data of the current route
    trav       : travel times of this route
    trav_sort  : sorted travel times of this route
    '''
    from copy import deepcopy
    trav        = deepcopy(travel)
    trav_sort   = deepcopy(travel_sort)
    station     = get_station(route_data)
    propose     = [station,]
    to_go       = []
    time_passed = 0.0
    for stops in route_data['stops']:
        to_go.append(stops)
    
    def remove_from_all(stop: str):
        'remove the specified stop from all mentions'
        to_go.remove(stop)

        for stops in trav:
            if (trav[stops].__contains__(stop)):
                trav[stops].__delitem__(stop)
            if (trav_sort[stops].__contains__(stop)):
                trav_sort[stops].remove(stop)

    current = station
    remove_from_all(station)
    while len(to_go)>0:
        scores = {}
        
        first   = trav_sort[current][:1]
        best    = first[0]
        
        for next_stop in trav_sort[current][:3]:
            test_pred = create_dmatrix(route_data, travel, current, next_stop, time_passed)
            pred      = exp(model.predict(test_pred)) - 1 # returns an 1*1 array

            scores[next_stop+'_self'] = pred[0]

            sum = 0
            scores[next_stop] = {}
            for future_stop in trav_sort[next_stop][:3]:
                test_pred = create_dmatrix(route_data, travel, current, next_stop, time_passed)
                pred      = exp(model.predict(test_pred)) - 1 # returns an 1*1 array
                
                scores[next_stop][future_stop] = pred[0]
            
            scores[next_stop+'_total'] = scores[next_stop+'_self'] + sum  # the final score to use for considering which to choose
            if scores[next_stop+'_total']>scores[best+'_total']:
                best = next_stop
        
        propose.append(best)
        time_passed += travel[current][best]
        current = best
        remove_from_all(current)
    
    return propose


# %%
# Create all the proposal
logging.info('Start predicting')
result = {}
for route in prediction_routes:
    logging.info('Start prediction for ' + route)
    result[route] = create_proposal(prediction_routes[route], travel_times[route], travel_times_sorted[route])

logging.info('Prediction done')


# %%
# Write output data
logging.info('Outputting the result')
result_path = path.join(output_path, 'proposed_sequences.json')
def create_output(result: dict) -> dict:
    out = {}
    for route in result:
        out[route]             = {}
        out[route]['proposed'] = {}

        for i in range(0, len(result[route]) - 1):
            stop                         = result[route][i]
            out[route]['proposed'][stop] = i
    
    return out

output = create_output(result)

with open(result_path, 'w') as out_file:
    json.dump(output, out_file, indent=4)

logging.info('Done')
print('Done!')

# %%
