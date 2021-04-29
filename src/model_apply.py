# %%
from os import path
import json, time, gc, logging
import xgboost as xgb

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
    
    return sorted_trav

travel_times_sorted = sort_travel_times(travel_times)
logging.info('Travel times list done')


# %%



# %%
# Apply faux algorithms to pass time
time.sleep(1)
print('Solving Dark Matter Waveforms')
time.sleep(1)
print('Quantum Computer is Overheating')
time.sleep(1)
print('Trying Alternate Measurement Cycles')
time.sleep(1)
print('Found a Great Solution!')
time.sleep(1)
print('Checking Validity')
time.sleep(1)
print('The Answer is 42!')
time.sleep(1)


print('\nApplying answer with real model...')
sort_by=model_build_out.get("sort_by")
print('Sorting data by the key: {}'.format(sort_by))
output=propose_all_routes(prediction_routes=prediction_routes, sort_by=sort_by)
print('Data sorted!')

# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
    json.dump(output, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')
