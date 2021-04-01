import json

with open("data/model_apply_inputs/new_package_data.json", "r") as package_data:
    package_dat = json.load(package_data)

with open("data/model_apply_inputs/new_route_data.json", "r") as route_data:
    route_dat = json.load(route_data)

def size_each(package):
    for route in package:
        print("Route :", route)
        vol_tot = 0

        for stop in package[route]:
            for package_id in package[route][stop]:
                vol = 1
                for dim in package[route][stop][package_id]["dimensions"]:
                    vol *= package[route][stop][package_id]["dimensions"][dim]
                
                vol_tot += vol
        
        print("Total volume:", vol_tot, "\b\n")
        package[route]["capacity_use"] = vol_tot
    return 0

def compare(package, route_dat):
    found_tot = 0
    for route in package:
        capacity = route_dat[route]["executor_capacity_cm3"]
        usage = package[route]["capacity_use"]
        
        if usage >= capacity:
            print("Found in", route)
            found_tot += 1
    
    print("Found:", found_tot)
    return 0



size_each(package_dat)
compare(package_dat, route_dat)
