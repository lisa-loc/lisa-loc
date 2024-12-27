import math
import os
import mgrs
import utm
import open3d as o3d
import numpy as np

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def check_if_image_exists(mapbox_image_folder, x=None, y=None, zoom=None, filename=None):
    if filename is not None and x is None and y is None and zoom is None:
        return os.path.isfile(f"{mapbox_image_folder}/{filename}")
    elif x is not None and y is not None and zoom is not None and filename is None:
        return os.path.isfile(f"{mapbox_image_folder}/{x}_{y}_{zoom}.jpg")
    else:
        return False
    
def check_if_pcd_exists(pcd_folder, x, y, zoom):
    return os.path.isfile(f"{pcd_folder}/{x}_{y}_{zoom}.pcd")

def get_x_y_zoom_from_filename(filename):
    x, y, zoom = filename.split(".")[0].split("_")
    return int(x), int(y), int(zoom)

def get_mgrs(x, y, zoom):
    lat, long = num2deg(x, y, zoom)
    mgrs_obj = mgrs.MGRS()
    utm_obj = utm.from_latlon(lat, long)
    mgrs_string = mgrs_obj.toMGRS(lat, long, MGRSPrecision=5)
    # print(mgrs_string)
    mgrs_x, mgrs_y = mgrs_string[5:10], mgrs_string[10:]

    # print(mgrs_x, mgrs_y, utm_obj[0], utm_obj[1])
   
    mgrs_x = int(mgrs_x) + (utm_obj[0] - int(utm_obj[0])) 
    mgrs_y = int(mgrs_y) + (utm_obj[1] - int(utm_obj[1]))

    return mgrs_x, mgrs_y

def get_mgrs_from_lat_long(lat, long):
    mgrs_obj = mgrs.MGRS()
    utm_obj = utm.from_latlon(lat, long)
    mgrs_string = mgrs_obj.toMGRS(lat, long, MGRSPrecision=5)
    # print(mgrs_string)
    mgrs_x, mgrs_y = mgrs_string[5:10], mgrs_string[10:]
   
    mgrs_x = int(mgrs_x) + (utm_obj[0] - int(utm_obj[0])) 
    mgrs_y = int(mgrs_y) + (utm_obj[1] - int(utm_obj[1]))

    return mgrs_x, mgrs_y

def load_map_pcd1(x,y,zoom, folder_path):
    map_points = None
    # print(x,y,zoom)
    for i in range(-1,2):
        inner_x = i*10 + x
        for j in range(-1,2):
            inner_y = j*10 + y
            pcd = o3d.io.read_point_cloud(f"{folder_path}/{inner_x}_{inner_y}_{zoom}.pcd")

            if map_points is None:
                map_points = np.asarray(pcd.points)
            else:
                map_points = np.concatenate((map_points, np.asarray(pcd.points)), axis=0)

    return map_points




