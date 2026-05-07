'''
from pyproj import Transformer

# Initialize transformers (one-time setup)
ecef_transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)  # WGS84 (3D) → ECEF
lla_transformer = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)    # ECEF → WGS84 (3D)

def geodetic_to_ecef(lat, lon, alt):
    """Convert WGS84 (lat, lon, alt) to ECEF (x, y, z) in meters"""
    return ecef_transformer.transform(lon, lat, alt)  # Note: lon, lat order!

def ecef_to_geodetic(x, y, z):
    """Convert ECEF (x, y, z) to WGS84 (lat, lon, alt)"""
    return lla_transformer.transform(x, y, z)  # Returns (lon, lat, alt)

lat, lon, alt = 16.37239, 48.20841, 0
x1, y1, z1 = ecef_transformer.transform(lon, lat, alt)  # -> Proper ECEF coordinates
print("lonlat:",x1, y1, z1)

'''

import numpy as np
import random
import math

# WGS84 Ellipsoid parameters
a = 6378137.0
f = 1.0 / 298.257223563
b = (1.0 - f) * a
e = math.sqrt(a**2 - b**2) / a

def deg2rad(deg):
    return deg * math.pi / 180.0

def sample_normal_distribution(mean, std_dev):
    return random.gauss(mean, std_dev)

def normalize_angles(angle):
    angle = (angle + math.pi) % (2.0 * math.pi)
    if angle < 0:
        angle += 2.0 * math.pi
    return angle - math.pi

def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), math.sin(theta)],
        [0, -math.sin(theta), math.cos(theta)]
    ])

def Ry(theta):
    return np.array([
        [math.cos(theta), 0, -math.sin(theta)],
        [0, 1, 0],
        [math.sin(theta), 0, math.cos(theta)]
    ])

def Rz(theta):
    return np.array([
        [math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

def lla_to_ecef(points_lla):
    points_ecef = []
    for point in points_lla:
        lon = math.radians(point[0])
        lat = math.radians(point[1])
        alt = point[2]
        N = a / math.sqrt(1.0 - (e * math.sin(lat))**2)
        x = (N + alt) * math.cos(lat) * math.cos(lon)
        y = (N + alt) * math.cos(lat) * math.sin(lon)
        z = (N * (1.0 - e**2) + alt) * math.sin(lat)
        points_ecef.append([x, y, z])
    return points_ecef

def ecef_to_enu(points_ecef, ref_lla):
    points_enu = []
    lon = deg2rad(ref_lla[0])
    lat = deg2rad(ref_lla[1])
    alt = ref_lla[2]
    
    ref_ecef = lla_to_ecef([ref_lla])[0]
    
    R = Rz(math.pi / 2.0) @ Ry(math.pi / 2.0 - lat) @ Rz(lon)
    
    for point in points_ecef:
        relative = np.array([point[0] - ref_ecef[0], point[1] - ref_ecef[1], point[2] - ref_ecef[2]])
        enu = R @ relative
        points_enu.append([enu[0], enu[1], enu[2]])
    
    return points_enu

def quaternion_to_rotation_matrix(x, y, z, w):
    """Convert quaternion to 3x3 rotation matrix (Hamilton convention, ROS standard)"""
    # Normalize quaternion (important for valid rotation!)
    norm = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    # Compute rotation matrix
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# Your IMU quaternion
q_x, q_y, q_z, q_w = -0.0098595, 0.0009070, 0.4769293, 0.8788859

R = quaternion_to_rotation_matrix(q_x, q_y, q_z, q_w)
print("Rotation Matrix:\n", R)

def lla_to_enu(points_lla, ref_lla):
    points_ecef = lla_to_ecef(points_lla)
    return ecef_to_enu(points_ecef, ref_lla)

# Example: Convert LLA to ENU
points_lla = [
    [11.438214183883435, 48.75746174362819, 414.59326171875],  # [longitude, latitude, altitude] in degrees and meters
    [11.438265986628801, 48.757515292548064, 414.5531921386719]
]
ref_lla = points_lla[0]  # Reference = first point

enu_coords = lla_to_enu(points_lla, ref_lla)
ecef_coords = lla_to_ecef(points_lla)
ecef_enu_coords = ecef_to_enu(ecef_coords, ref_lla)
print("ENU Coordinates:", enu_coords)
print("ECEF Coordinates:", ecef_coords)
print("ECEF_ENU Coordinates:", ecef_enu_coords)

east = 3.8091134004463667
north = 5.955298345932568
horizontal_distance = math.sqrt(east**2 + north**2)
print(horizontal_distance)
