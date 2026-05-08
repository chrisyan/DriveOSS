#Access the specific data form gps and imu files # DOne
#Convert GPS coordinates (latitude, longitude, altitude) to Cartesian coordinates (x, y, z)

#Convert the IMU quaternion (x, y, z, w) to a rotation matrix
#Combine them into a 3×4 transformation matrix (rotation + translation)
#Format the output as specified

import re
import os
from pyproj import Transformer, CRS
import numpy as np
import math

def parse_gps(file_path):

    with open(file_path,'r') as f:
        gps_data = f.read() 
    
    lat = float(re.search(r'Latitude: ([\d.]+)',gps_data).group(1))
    lon = float(re.search(r'Longitude: ([\d.]+)',gps_data).group(1))
    alt = float(re.search(r'Altitude: ([\d.]+)',gps_data).group(1))

    return lat, lon, alt

def parse_gps_directory(directory_path, return_enu=False):

    gps_data = {}
    enu_data = {}

    for gps_file in sorted(os.listdir(directory_path)):
        if gps_file.endswith('.txt'):
            try:
                gps_path = os.path.join(directory_path, gps_file)
                lat, lon, alt = parse_gps(gps_path)
                x, y, z = latlon_xyz(lat,lon,alt)
                gps_data[gps_file] = (x, y, z)
            except Exception as e:
                print(f"Skipping {gps_file}: {str(e)}")
    
    if return_enu:
        if not gps_data:
            return {}
        
        # Use first point as reference
        first_file = min(gps_data.keys())
        ref_lla = parse_gps(os.path.join(directory_path, first_file))
        
        for file_name, ecef in gps_data.items():
            enu_data[file_name] = ecef_to_enu(ref_lla, ecef)
        return enu_data
    
    return gps_data

def parse_imu(file_path):

    with open(file_path,'r') as f:
        imu_data = f.read()

    match = re.search(r'Orientation:\s*geometry_msgs\.msg\.Quaternion\(x=([\d.-]+),\s*y=([\d.-]+),\s*z=([\d.-]+),\s*w=([\d.-]+)\)', imu_data)

    imu_x = float(match.group(1))
    imu_y = float(match.group(2))
    imu_z = float(match.group(3))
    imu_w = float(match.group(4))
    
    return imu_x, imu_y, imu_z, imu_w

def parse_imu_directory(directory_path):

    imu_data = {}

    for imu_file in sorted(os.listdir(directory_path)):
        if imu_file.endswith('.txt'):
            try:
                imu_path = os.path.join(directory_path, imu_file)
                #imu_x, imu_y, imu_z, imu_w = parse_imu(imu_path)
                #imu_data[imu_file] = (quaternion_to_rotation_matrix(imu_x, imu_y, imu_z, imu_w))
                imu_data[imu_file] = parse_imu(imu_path)
            except Exception as e:
                print(f"Skipping {imu_file}: {str(e)}")

    return imu_data

def latlon_xyz(lat,lon,alt):
    xyz_transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    return xyz_transformer.transform(lon, lat, alt)

def ecef_to_enu(ref_lla, target_ecef):
    """
        ref_lla: Reference point as [lat, lon, alt] in degrees/meters
        target_ecef: Target point in ECEF [x, y, z] in meters
    """
    ref_ecef = np.array(latlon_xyz(*ref_lla))
    
    lat_rad = math.radians(ref_lla[0])
    lon_rad = math.radians(ref_lla[1])
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # ECEF to ENU rotation matrix
    '''
    https://docs.fixposition.com/fd/converting-from-ecef-to-enu-local-frame
    '''
    R = np.array([
        [-sin_lon,          cos_lon,          0      ],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
        [ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat]
    ])
    
    # Calculate ENU coordinates
    delta = np.array(target_ecef) - ref_ecef
    return tuple(R @ delta)  # (east, north, up)

def quaternion_to_rotation_matrix(x,y,z,w):
    # norm of the quaternion
    norm = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    Q2R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return Q2R

#need to modify the imu data as first is not considered as reference and include time stamp
def generate_poses(gps_folder, imu_folder, output_file="poses.txt"):
    # Get ENU coordinates (first GPS is reference)
    enu_results = parse_gps_directory(gps_folder, return_enu=True)
    imu_results = parse_imu_directory(imu_folder)
    
    # Find common timestamps
    common_files = sorted(set(enu_results.keys()) & set(imu_results.keys()))
    
    if not common_files:
        raise ValueError("No matching GPS/IMU files found!")
    
    # Get first IMU orientation as reference
    first_file = common_files[0]
    ref_quat = imu_results[first_file]
    ref_R = quaternion_to_rotation_matrix(*ref_quat)
    #print("ref_R",ref_R)
    
    with open(output_file, 'w') as f:
        
        for filename in common_files:
            try:
                east, north, up = enu_results[filename]
                
                current_quat = imu_results[filename]
                current_R = quaternion_to_rotation_matrix(*current_quat)
                relative_R = ref_R.T @ current_R  
                
                # Create 3x4 pose matrix
                pose_matrix = np.column_stack((relative_R, [east, north, up]))
                
                pose_line = ' '.join([f"{val:.15f}" for row in pose_matrix for val in row])
                f.write(pose_line + '\n')
                #print(pose_line)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":

    gps_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/200/gps/000196.txt"
    gps_folder_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/200/gps"
    imu_file_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/200/imu/000196.txt"
    imu_folder_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/data/test/200/imu"
    
    '''
    gps_results = parse_gps_directory(gps_folder_path)
    for gps_file in sorted(gps_results.keys()):
        gps_x, gps_y, gps_z = gps_results[gps_file]
        #print(f"{gps_file}: GPS_X={gps_x:.7f}, GPS_Y={gps_y:.7f}, GPS_Z={gps_z:.2f}")
    
    
    enu_results = parse_gps_directory(gps_folder_path, return_enu=True)

    for filename, (east, north, up) in enu_results.items():
        print(f"{filename}: East={east:.2f}m, North={north:.2f}m, Up={up:.2f}m")
   

    
    imu_results = parse_imu_directory(imu_folder_path)
    #print("imu_data:", imu_results)
    
    #for imu_file in sorted(imu_results.keys()):
        #x, y, z, w = imu_results[imu_file]
        #x, y, z = imu_results[imu_file]
        #print(f"{imu_file}: Quaternion_X={x:.7f}, Quaternion_Y={y:.7f}, Quaternion_Z={z:.7f}, Quaternion_W={w:.7f}")
        #print(f"{imu_file}: Quaternion_X={x:.7f}, Quaternion_Y={y:.7f}, Quaternion_Z={z:.7f}")
    '''
    generate_poses(gps_folder_path, imu_folder_path)

    