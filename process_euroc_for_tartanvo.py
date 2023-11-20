import numpy as np
import cv2
from pathlib import Path
import yaml
import math


euroc_folders = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

data_root = "/home/akashsharma/datasets/EuRoC"

out_folder = "/home/akashsharma/workspace/cloned/tartanvo/data/EuRoC"


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


for folder in euroc_folders:
    camera_folder = Path(data_root) / folder / "mav0/cam0/data"
    pose_file = Path(data_root) / folder / "mav0/state_groundtruth_estimate0/data.csv"
    poses = np.loadtxt(pose_file, delimiter=",")
    poses = dict([(float(pose[0]) * 1e-9, pose[1:]) for pose in poses])

    camera_sensor_file = Path(data_root) / folder / "mav0/cam0/sensor.yaml"
    sensor_cfg = None
    out_camera_folder = Path(out_folder) / folder
    with open(camera_sensor_file) as sensor_file:
        sensor_file = sensor_file.read().split("\n", 1)[1]
        sensor_cfg = yaml.safe_load(sensor_file)

    fu, fv, cu, cv = sensor_cfg["intrinsics"]
    intrinsics = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.array(sensor_cfg["distortion_coefficients"], np.float32)
    image_timestamps = [
        [np.int64(image_name.stem) * 1e-9, image_name]
        for image_name in camera_folder.iterdir()
    ]
    image_timestamps = dict(image_timestamps)
    print("Matching timestamps...")
    matches = associate(poses, image_timestamps, 0.0, 0.02)
    print("Done")
    pose_lines = []
    print("Undistorting images...")
    out_camera_folder.mkdir(parents=True, exist_ok=True)
    (out_camera_folder / "image_left").mkdir(parents=True, exist_ok=True)
    for idx, (a, b) in enumerate(matches):
        pose_line = poses[a]
        image_name = image_timestamps[b]
        img = cv2.imread(str(image_name))
        img = cv2.undistort(img, intrinsics, distortion_coeffs)
        cv2.imwrite(str(out_camera_folder / f"image_left/{idx:06d}.png"), img)
        pose_lines.append(pose_line)
    print("Done")

    pose_lines = np.array(pose_lines)
    np.savetxt(out_camera_folder / "pose_left.txt", pose_lines)
