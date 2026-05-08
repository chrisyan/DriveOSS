import os
import struct
import numpy as np


def read_bin_file(bin_file_path):
    point_cloud = []
    with open(bin_file_path, 'rb') as f:
        while True:
            data = f.read(16)  # 4 * 4 bytes for x, y, z, intensity
            if not data:
                break
            point = struct.unpack('ffff', data)
            point_cloud.append(point)
    return np.array(point_cloud, dtype=np.float32)


def write_pcd_file_ascii(pcd_file_path, point_cloud):
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {len(point_cloud)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(point_cloud)}
DATA ascii
"""

    with open(pcd_file_path, 'w') as f:
        f.write(header)
        for point in point_cloud:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")


def bin_to_pcd_ascii(bin_file_path, pcd_file_path):
    point_cloud = read_bin_file(bin_file_path)
    write_pcd_file_ascii(pcd_file_path, point_cloud)


def convert_folder_ascii(bin_folder, pcd_folder):
    if not os.path.exists(pcd_folder):
        os.makedirs(pcd_folder)

    for filename in os.listdir(bin_folder):
        if filename.endswith('.bin'):
            bin_file_path = os.path.join(bin_folder, filename)
            pcd_file_path = os.path.join(pcd_folder, filename.replace('.bin', '.pcd'))
            bin_to_pcd_ascii(bin_file_path, pcd_file_path)
            print(f"Converted {bin_file_path} to {pcd_file_path}")


bin_folder = '/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/velodyne'
pcd_folder = '/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/data/stu_dataset/validation/141/lidar_pcd_ascii'
convert_folder_ascii(bin_folder, pcd_folder)