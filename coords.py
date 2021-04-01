import json

def to_csv(file, new_name):
    with open(file, "r") as source_file:
        source = json.load(source_file)

    f = open(new_name, "w", encoding = 'utf-8')
    f.write("Route,Station Code,Stops,lat,lng\n")

    for route in source:
        station = source[route]["station_code"]
        write_stat = route + "," + station + ","
        
        for stop in source[route]["stops"]:
            lat = source[route]["stops"][stop]["lat"]
            lng = source[route]["stops"][stop]["lng"]
            f.write(write_stat + stop + "," + str(lat) + "," + str(lng) + "\n")
    f.close()

to_csv("data/model_apply_inputs/new_route_data.json", "new_coord.csv")
to_csv("data/model_build_inputs/route_data.json", "full_coord.csv")
