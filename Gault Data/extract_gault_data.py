
# Current Progress (Initiate Gault DataSet Preprocessing):

# Oxford dataset is fully done (voxelization, loop-closure labels, PyTorch Dataset/DataLoader) 
# across all four sites. Now I am starting Gault DataSet preprocessing . 

# I received the Oxford data, very nicely organized from their website.
# Each LiDAR scan was already a sepaate file, so I could immediately start processing it.


# Next tasks:

# My supervisor at MRL, William, collected real forest data at Gault Nature Reserve Park in Quebec
# with a robot, but that data needs the same processing before I can use it.
# But The LiDAR scans are not separated yet and everything is packed inside an enormous 
# 62 GB .mcap file.

# In plain  words, the process is basically:
# Robot records of Gault forest collected by Willaim → huge .mcap file → 
# extract LiDAR scans → many small .pcd files → process them like Oxford DataSet
#  AND extract robot-poses too -> csv file






# The script turns William's one big Gault recording into the same two simple things that the
# Oxford data already has. Right now everything: LiDAR scans, camera images, IMU readings, 
# robot position, is bundled together inside one mcap file, which none of my existing 
# code knows how to read. 
# 
# So the script opens that file and does two separate jobs: 
# 
# 1) First, it goes through every LiDAR scan and saves each one as its own .pcd file, exactly like the
# individual scan files we already have for Wytham, Stein-am-Rhein, Forest of Dean and Evo. 

# 2) Second, it goes through every robot-position reading (IN the KISS-ICP output William already
# computed) and saves it as a CSV file with x, y, z coordinates, same columns as Oxford
# slam_poses.csv files. Once both of these exist for Gault, the voxelization, loop-closure 
# labeling, and Dataset/DataLoader code can run on Gault without changing a single line
# of them. To those scripts, Gault just looks like a fifth site.


# NOTE:
# My script has two jobs: get the LiDAR scans and get the robot's positions.
# The LiDAR part worked perfectly — all 520 scans were extracted. The problem is with 
# the robot positions, called odometry. I originally tried to use pointcloudset for
# both jobs, but pointcloudset is designed only for LiDAR point clouds, not 
# robot-position data. So when it tried to read the odometry, it failed. 
# The solution was simple: keep using pointcloudset for the LiDAR scans, 
# but use another tool called rosbags to read the robot positions. Then I can
# save those positions into a CSV file, just like the position file I had for Oxford.


# For the results, we can think that this script gives us two separate notebooks:

# Think of two separate notebooks.
# Notebook 1 (scans): a pile of photos, each labeled with the time it was taken. "Photo at 10:00:00.1", "Photo at 10:00:00.2", and so on.
# Notebook 2 (poses): a list of positions, each with the time it was recorded. "10:00:00.1: robot at x=5, y=2", and so on.
# Our script made both notebooks from the bag.
# For Oxford we didn't need this step, because its table already said "this position belongs to this photo."




# Accordign to the metadata.yaml file, the small 6GB .mcap file I have contains 160,734 messages in total, and each topic is a stream of one kind of reading.

# LiDAR (what we need): /livox/lidar has 3,516 scans. /livox/imu has 6,684 motion readings.
# Camera (the likely cause of the huge size): colour images (3,495), depth images (about 3,470), and stereo point clouds (3,495), plus a camera IMU.
# Robot platform: wheel odometry (odom/filtered, 17,037), joystick commands, joint states, battery and emergency stop.
# RTAB-Map (a camera-based mapping system): its own odometry, map graph, map path and cloud map.
# KISS-ICP outputs: /kiss/odometry (3,504 poses), /kiss/frame, /kiss/local_map and /kiss/keypoints.
# Frame transforms and IMU: /tf, /tf_static, /imu/data_raw.
# Our script only reads two of these: /livox/lidar for the scans and a pose topic such as /kiss/odometry for the poses. The rest, mostly images and depth, is what makes the file 62 GB.



# Importantly, LiDAR (what we need): /livox/lidar has 3,516 scans.
# KISS-ICP outputs: /kiss/odometry (3,504 poses)
# These were inside the small 6gb sample of the large .mcap file colelcted at the Gault Forest by the robot.
# This script saves the scans as .pcd files and the poses as a CSV (slam_poses.csv).


import argparse
import os
import signal
from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore

# The Gault bags are Clearpath W200 recordings (Ouster LiDAR), not the Livox/KISS-ICP
# topics of the sample bag. Override any of these from the command line.
DEFAULT_LIDAR_TOPIC = "/w200_0110/sensors/lidar3d_0/points"
DEFAULT_ODOMETRY_TOPIC = "/w200_0110/platform/odom/filtered"

POINTFIELD_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

stop_requested = False


def request_stop(signum, frame):
    """Ctrl-C / SIGTERM: finish the scan being written, then exit cleanly. Rerun to resume."""
    global stop_requested
    if stop_requested:
        raise KeyboardInterrupt
    stop_requested = True
    print("\nStop requested: finishing current scan (press Ctrl-C again to force quit). "
          "Rerun the same command to resume.")


def pointcloud2_to_xyz(message):
    """Decode a sensor_msgs/PointCloud2 into an (N, 3) float64 array, dropping NaN/inf points."""
    fields = {f.name: f for f in message.fields}
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [("<" if not message.is_bigendian else ">") + POINTFIELD_DTYPES[fields[n].datatype]
                    for n in "xyz"],
        "offsets": [fields[n].offset for n in "xyz"],
        "itemsize": message.point_step,
    })
    n_points = message.width * message.height
    raw = np.frombuffer(bytes(message.data), dtype=dtype, count=n_points)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def last_extracted_timestamp(output_folder):
    """Timestamp (ns) of the newest cloud_<ns>.pcd already on disk, or None."""
    stamps = [int(f[6:-4]) for f in os.listdir(output_folder)
              if f.startswith("cloud_") and f.endswith(".pcd") and f[6:-4].isdigit()]
    return max(stamps) if stamps else None


def extract_all_scans(bag_folder_path, lidar_topic_name, output_folder):
    """
    Streams the LiDAR topic straight out of the bag (no giant intermediate dataset),
    converts each scan to Open3D and writes cloud_<timestamp>.pcd. Files are written to a
    temp name and renamed, so a killed run never leaves a half-written scan. On rerun, the
    newest existing file tells us where to seek, so work resumes where it stopped.
    """
    os.makedirs(output_folder, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    last_done = last_extracted_timestamp(output_folder)
    start = last_done + 1 if last_done is not None else None

    with Rosbag2Reader(Path(bag_folder_path)) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic_name]
        if not connections:
            topics = sorted(c.topic for c in reader.connections)
            raise SystemExit(f"LiDAR topic {lidar_topic_name} not in bag. Available topics:\n  "
                             + "\n  ".join(topics))
        total = sum(c.msgcount for c in connections)
        already = len(os.listdir(output_folder))
        if start is not None:
            print(f"Resuming after timestamp {last_done} ({already} scans already on disk)")

        count = already
        for connection, timestamp, raw_data in reader.messages(connections=connections, start=start):
            message = typestore.deserialize_cdr(raw_data, connection.msgtype)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pointcloud2_to_xyz(message))

            final_path = os.path.join(output_folder, f"cloud_{timestamp}.pcd")
            tmp_path = final_path[:-4] + ".tmp.pcd"
            o3d.io.write_point_cloud(tmp_path, cloud)
            os.replace(tmp_path, final_path)
            count += 1

            if count % 100 == 0 or count == total:
                print(f"Extracted {count}/{total} scans", flush=True)
            if stop_requested:
                print(f"Stopped at {count}/{total} scans. Rerun to resume.")
                return False

    print(f"\nFinished extracting {count} scans to {output_folder}")
    return True


def extract_odometry_to_csv(bag_folder_path, odometry_topic_name, output_csv_path):
    """Reads nav_msgs/Odometry messages and saves timestamp, x, y, z, qx, qy, qz, qw to CSV
    (same columns as Oxford slam_poses.csv). Skipped if the CSV already exists."""
    if os.path.exists(output_csv_path):
        print(f"{output_csv_path} already exists, skipping odometry")
        return

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    with Rosbag2Reader(Path(bag_folder_path)) as reader:
        connections = [c for c in reader.connections if c.topic == odometry_topic_name]
        if not connections:
            topics = sorted(c.topic for c in reader.connections)
            raise SystemExit(f"Odometry topic {odometry_topic_name} not in bag. Available topics:\n  "
                             + "\n  ".join(topics))
        for connection, timestamp, raw_data in reader.messages(connections=connections):
            pose = typestore.deserialize_cdr(raw_data, connection.msgtype).pose.pose
            rows.append({
                "timestamp": timestamp,
                "x": pose.position.x, "y": pose.position.y, "z": pose.position.z,
                "qx": pose.orientation.x, "qy": pose.orientation.y,
                "qz": pose.orientation.z, "qw": pose.orientation.w,
            })

    tmp_path = output_csv_path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_csv_path)
    print(f"Saved {len(rows)} poses to {output_csv_path}")


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Extract LiDAR scans and odometry poses from a ROS2 bag (stoppable/resumable)."
    )
    parser.add_argument("--bag-folder", required=True,
                        help="Folder containing the .mcap file and its metadata.yaml")
    parser.add_argument("--output-folder", default=".",
                        help="Folder where individual_clouds/ and slam_poses.csv will be created")
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC,
                        help="PointCloud2 topic to extract (default: %(default)s)")
    parser.add_argument("--odometry-topic", default=DEFAULT_ODOMETRY_TOPIC,
                        help="nav_msgs/Odometry topic for poses (default: %(default)s)")
    parser.add_argument("--poses-csv-name", default="slam_poses.csv",
                        help="File name for the pose CSV inside the output folder (default: %(default)s)")
    parser.add_argument("--skip-odometry", action="store_true", help="Only extract LiDAR scans")
    parser.add_argument("--skip-lidar", action="store_true", help="Only extract odometry")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    site_folder = arguments.output_folder
    os.makedirs(site_folder, exist_ok=True)

    if not arguments.skip_odometry:
        print("Loading odometry poses from ROS2 bag...")
        extract_odometry_to_csv(arguments.bag_folder, arguments.odometry_topic,
                                os.path.join(site_folder, arguments.poses_csv_name))

    if not arguments.skip_lidar:
        print("\nLoading LiDAR scans from ROS2 bag...")
        finished = extract_all_scans(arguments.bag_folder, arguments.lidar_topic,
                                     os.path.join(site_folder, "individual_clouds"))
        print("\nExtraction complete." if finished else "\nExtraction paused.")
