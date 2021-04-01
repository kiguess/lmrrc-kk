import json


def from_folder(folder: str, station:str="DCH4"):
    with open(folder+"route_data.json", "r") as dat:
        route_dat = json.load(dat)
    with open(folder+"actual_sequences.json", "r") as act:
        act_seq = json.load(act)
    with open(folder+"package_data.json", "r") as pack:
        package = json.load(pack)
    
    r = open("bui_csv/route_data.csv", "w", encoding='utf-8')
    a = open("bui_csv/actual_sequences.csv", "w", encoding='utf8')
    p = open("bui_csv/package_data.csv", "w", encoding='utf-8')
    
    r.write("Route,Station Code,date,depart_time,executor_capacity,score,Stops,lat,long,type,zone\n")
    a.write("Route,stop,seq_num\n")
    p.write("Route,stop,package,status,start_time,end_time,planned_service_time,depth,height,width\n")

    def dump_route_data(dat, route):
        station = dat[route]["station_code"]
        date = dat[route]["date_YYYY_MM_DD"]
        depart = dat[route]['departure_time_utc']
        capacity = dat[route]['executor_capacity_cm3']
        score = dat[route]['route_score']
        write_stat = route + "," + station + "," + date + ',' + depart + ',' + str(capacity) + ',' + score + ','
        
        for stop in dat[route]["stops"]:
            lat = dat[route]["stops"][stop]["lat"]
            lng = dat[route]["stops"][stop]["lng"]
            stype = dat[route]["stops"][stop]["type"]
            zone = dat[route]["stops"][stop]["zone_id"]
            r.write(write_stat + stop + "," + str(lat) + "," + str(lng) + ',' + stype + ',' + str(zone) + "\n")
    
    def dump_actual_seq(dat, route):
        write_stat = route + ','

        for stops in dat[route]["actual"]:
            seq = dat[route]["actual"][stops]
            a.write(write_stat + stops + "," + str(seq) + '\n')
    
    def dump_package_dat(dat, route):
        for stops in dat[route]:
            write_stat = route + ',' + stops + ','
            for packages in dat[route][stops]:
                status = dat[route][stops][packages]["scan_status"]
                start_time = dat[route][stops][packages]["time_window"]["start_time_utc"]
                end_time = dat[route][stops][packages]["time_window"]["end_time_utc"]
                service_time = dat[route][stops][packages]["planned_service_time_seconds"]
                depth = dat[route][stops][packages]["dimensions"]["depth_cm"]
                height = dat[route][stops][packages]["dimensions"]["height_cm"]
                width = dat[route][stops][packages]["dimensions"]["width_cm"]
                package_dat = packages + ',' + status + ',' + str(start_time) + ',' + str(end_time) + ',' + str(service_time) + ','
                package_dat += str(depth) + ',' + str(height) + ',' + str(width)
                p.write(write_stat + package_dat + '\n')

    for route in route_dat:
        if route_dat[route]["station_code"] != station:
            continue
        else:
            dump_route_data(route_dat, route)
            dump_actual_seq(act_seq, route)
            dump_package_dat(package, route)

    r.close()
    a.close()
    p.close()

from_folder("data/model_build_inputs/", "DCH4")