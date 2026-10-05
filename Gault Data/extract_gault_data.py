
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
# Robot records of Gault forest collected by Willaim → huge .mcap file → extract LiDAR scans → many small .pcd files → process them like Oxford DataSet






# The script turns William's one big Gault recording into the same two simple things that the
# Oxford data already has. Right now everything: LiDAR scans, camera images, IMU readings, 
# robot position, is bundled together inside one mcap file, which none of my existing 
# code knows how to read. So the script opens that file and does two separate jobs: first, it 
# goes through every LiDAR scan and saves each one as its own .pcd file, exactly like the
# individual scan files we already have for Wytham, Stein-am-Rhein, Forest of Dean and Evo. 
# Second, it goes through every robot-position reading (the KISS-ICP output William already
# computed) and saves it as a CSV file with x, y, z coordinates, same columns as Oxford
# slam_poses.csv files. Once both of these exist for Gault, your voxelization, loop-closure 
# labeling, and Dataset/DataLoader code can run on Gault without me changing a single line
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





import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d
from pointcloudset import Dataset
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore
 
 
 
 
def load_lidar_dataset(bag_folder_path, lidar_topic_name):
    """
    This function opens the ROS2 bag and loads only the LiDAR scan messages
    from it, using the pointcloudset library. A ROS2 bag is not just the
    single .mcap file by itself, it is a folder that holds the .mcap file
    together with a metadata.yaml file describing it, and pointcloudset
    needs to see both of these together to recognize and read the bag
    correctly. We do not load the whole bag into memory at once, since it
    also contains camera images, IMU readings, and other data we do not
    need for this step. By telling pointcloudset exactly which topic name
    holds the LiDAR data, in our case /livox/lidar, it reads only that
    stream and gives us back a Dataset object, which behaves like a list
    of individual LiDAR scans that we can loop through one at a time.
    
    This function Opens the big .mcap file and says: "Give me only the LiDAR scans." It ignores the other information.
    """
    
    lidar_dataset = Dataset.from_file(Path(bag_folder_path), topic=lidar_topic_name)
    return lidar_dataset
 
 
 
 
 
 
def convert_scan_to_open3d(pointcloudset_scan):
    """
    This function takes a single scan, as given to us by pointcloudset, and
    converts it into the Open3D point cloud format that the rest of our
    pipeline already understands and uses, since all of our Oxford work was
    built around Open3D point clouds. Internally, pointcloudset stores each
    scan's points as a table with x, y, and z columns, similar to a
    spreadsheet. We pull out just those three columns as plain numbers, and
    hand them to Open3D to build a proper point cloud object from them,
    which can then be saved to disk as a .pcd file exactly like our Oxford
    scans.
    
    A LiDAR scan initially contains thousands of points.
    And each point is a (x, y, z) 3D coordinate. This function 
    converts those numbers into a compatible 3D point cloud that Open3D can work with.
    """
    
    points_table = pointcloudset_scan.data
    xyz_points = points_table[["x", "y", "z"]].to_numpy(dtype=np.float64)
 
    open3d_cloud = o3d.geometry.PointCloud()
    open3d_cloud.points = o3d.utility.Vector3dVector(xyz_points)
 
    return open3d_cloud
 
 
 
 
 
 
 
def extract_all_scans(lidar_dataset, output_folder):
    """
    This function goes through every single LiDAR scan inside the loaded
    dataset, converts each one into an Open3D point cloud using the
    function above, and saves it to disk as its own .pcd file inside the
    given output folder. Each saved file is named using the scan's
    timestamp, which keeps every scan uniquely identifiable and keeps the
    naming style consistent with how Oxford's individual_clouds folder
    names its own files. This function also keeps a simple running count
    printed to the screen, since a Gault recording can contain thousands of
    scans, and it is useful to see progress while it runs rather than
    waiting in silence.
    
    This function goes through the LiDAR scans one by one, converts each scan, 
    and saves each as a separate .pcd file.
    
    So:
    big .mcap → scan1.pcd, scan2.pcd, scan3.pcd, ...
    
    """
    os.makedirs(output_folder, exist_ok=True)
 
    total_scans = len(lidar_dataset)
    extracted_count = 0
 
    for scan_index in range(total_scans):
        single_scan = lidar_dataset[scan_index]
        open3d_cloud = convert_scan_to_open3d(single_scan)
 
        scan_timestamp = single_scan.timestamp
        file_name = f"cloud_{scan_timestamp}.pcd"
        file_path = os.path.join(output_folder, file_name)
 
        o3d.io.write_point_cloud(file_path, open3d_cloud)
        extracted_count += 1
 
        if extracted_count % 100 == 0 or extracted_count == total_scans:
            print(f"Extracted {extracted_count}/{total_scans} scans")
 
    print(f"\nFinished extracting {extracted_count} scans to {output_folder}")
 
 
 
 
 
 
 
def load_odometry_dataset(bag_folder_path, odometry_topic_name):
    """
    This function opens the same ROS2 bag folder a second time, but this
    time it reads the odometry messages using the rosbags library instead
    of pointcloudset. We switch libraries here because pointcloudset is
    built only to understand point cloud messages, where each message
    carries a list of column names called fields, and our odometry
    messages are a completely different shape, a position and an
    orientation rather than a cloud of points, so pointcloudset cannot
    open them at all. The rosbags library is more general purpose and can
    decode any kind of ROS2 message, which is exactly what we need here.
    This function returns a plain Python list, where each entry pairs a
    message's timestamp with its decoded contents, one entry for every
    reading found on the /kiss/odometry topic.
    
    This function opens the .mcap but this time says: "Give me the robot's position information."
    This tells you where the robot was as it moved through the forest.
    
    """
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    odometry_readings = []
 
    with Rosbag2Reader(Path(bag_folder_path)) as reader:
        matching_connections = [
            connection for connection in reader.connections
            if connection.topic == odometry_topic_name
        ]
 
        for connection, timestamp, raw_data in reader.messages(connections=matching_connections):
            decoded_message = typestore.deserialize_cdr(raw_data, connection.msgtype)
            odometry_readings.append((timestamp, decoded_message))
 
    return odometry_readings
 
 
 
def extract_odometry_to_csv(odometry_readings, output_csv_path):
    """
    This function goes through every odometry reading we decoded and pulls
    out the robot's position, as x, y, and z coordinates, along with its
    orientation, as a quaternion made of qx, qy, qz, and qw values. We also
    record the timestamp of each reading. This mirrors exactly the same
    columns found in Oxford's slam_poses.csv file, which means once this
    CSV is saved, our existing scan spacing analysis and loop closure
    labeling scripts can read it directly, without needing any changes to
    understand a new file format.
    
    This functiom takes all those robot positions and puts them into one CSV file, 
    organized similarly to the Oxford slam_poses.csv.
    """
    pose_rows = []
 
    for timestamp, odometry_message in odometry_readings:
        position = odometry_message.pose.pose.position
        orientation = odometry_message.pose.pose.orientation
 
        pose_rows.append({
            "timestamp": timestamp,
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "qx": orientation.x,
            "qy": orientation.y,
            "qz": orientation.z,
            "qw": orientation.w,
        })
 
    poses_table = pd.DataFrame(pose_rows)
    poses_table.to_csv(output_csv_path, index=False)
 
    print(f"Saved {len(poses_table)} poses to {output_csv_path}")
 
 
 
 
 
def parse_command_line_arguments():
    """
    This function reads the two settings that change from one run to the
    next, which are where the ROS2 bag folder lives and where the results
    should be saved, straight from the command line. We do this so that
    the same script can be run on our small test sample and on William's
    full recording without anyone having to open the file and edit paths
    inside the code. The bag folder is the folder that holds the .mcap
    file together with its metadata.yaml. The output folder is where the
    individual_clouds folder and the slam_poses.csv file will be created,
    which gives Gault the same layout as each of the Oxford site folders.
    """
    parser = argparse.ArgumentParser(
        description="Extract Gault LiDAR scans and odometry poses from a ROS2 bag."
    )
    parser.add_argument(
        "--bag-folder",
        default="gault_sample_bag",
        help="Folder containing the .mcap file and its metadata.yaml",
    )
    parser.add_argument(
        "--output-folder",
        default=".",
        help="Folder where individual_clouds/ and slam_poses.csv will be created",
    )
    return parser.parse_args()
 
 
 
 
if __name__ == "__main__":
    arguments = parse_command_line_arguments()
 
    bag_folder_path = arguments.bag_folder
    site_folder = arguments.output_folder
    scans_folder = os.path.join(site_folder, "individual_clouds")
    poses_csv_path = os.path.join(site_folder, "slam_poses.csv")
    os.makedirs(site_folder, exist_ok=True)
 
    lidar_topic_name = "/livox/lidar"
    odometry_topic_name = "/kiss/odometry"
 
    print("Loading odometry poses from ROS2 bag...")
    odometry_readings = load_odometry_dataset(bag_folder_path, odometry_topic_name)
    extract_odometry_to_csv(odometry_readings, poses_csv_path)
 
    print("\nLoading LiDAR scans from ROS2 bag...")
    lidar_dataset = load_lidar_dataset(bag_folder_path, lidar_topic_name)
    extract_all_scans(lidar_dataset, scans_folder)
 
    print("\nExtraction complete.")
