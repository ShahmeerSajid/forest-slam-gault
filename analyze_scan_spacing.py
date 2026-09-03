

# This code is very important for my project. 

# This code is a simple analysis script that checks
# how far the robot moves between each LiDAR scan in each forest. It calculates the average
# and median distance between scans, the total distance the robot traveled, and how far the 
# robot ended from where it started. We need this to understand how the robot actually moved 
# and to help us choose a sensible number of scans for the loop-closure gap instead of 
# guessing. The results showed that the robot usually moved about 1.1–1.5 meters between 
# consecutive scans, which helped us decide on the settings for the next script.


# This code is an exploratory step, not something that produces final output 
# we use later. Its only job was to answer: as the robot walked, how much distance passed 
# between one scan and the next? 

# We needed this because our labeling rule (in generate_loop_closure.py) 
# needs a sensible number for "how many scans apart counts as a real revisit, not just the next 
# step." Without checking this, we'd have been guessing blindly. The result told us: consecutive 
# scans are usually ~1.1 to 1.5 meters apart, depending on the site.







import pandas as pd
import numpy as np
import os
import random
import open3d as o3d










def get_site_folders(project_root):
    """
    This function defines the four Oxford forest sites we are working with
    and builds the full folder path for each one, based on where the
    project lives on disk.
    """
    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
    site_paths = {}
    for site_name in site_names:
        site_paths[site_name] = os.path.join(project_root, site_name)
    return site_paths








def analyze_scan_spacing(site_folder, site_name):
    """
    This function loads a site's pose data and computes how far the robot
    moved between each consecutive pair of scans, meaning scan 1 to scan 2,
    scan 2 to scan 3, and so on. From these consecutive distances, we can
    work out the average distance between scans, which tells us how densely
    the trajectory was sampled. We also compute the total path length by
    adding up all of these consecutive distances, and we compute the
    straight-line distance between the very first and very last scan, which
    tells us whether the overall path returns close to where it started.
    Together, these numbers give us a grounded sense of scale for our site,
    which we can use to justify a distance threshold and a minimum sequence
    gap based on the robot's actual movement, rather than a guess.
    """
    poses_path = os.path.join(site_folder, "slam_poses.csv")

    if not os.path.isfile(poses_path):
        print(f"Skipping {site_name}: no slam_poses.csv found")
        return

    poses_table = pd.read_csv(poses_path)
    positions = poses_table[["x", "y", "z"]].values
    total_scans = len(poses_table)

    consecutive_distances = []
    for i in range(total_scans - 1):
        distance = np.linalg.norm(positions[i + 1] - positions[i])
        consecutive_distances.append(distance)

    consecutive_distances = np.array(consecutive_distances)
    average_spacing = np.mean(consecutive_distances)
    median_spacing = np.median(consecutive_distances)
    total_path_length = np.sum(consecutive_distances)

    start_to_end_distance = np.linalg.norm(positions[-1] - positions[0])

    scans_needed_to_cover_1_5m = 1.5 / average_spacing if average_spacing > 0 else float("inf")

    print(f"\nSite: {site_name}")
    print(f"Total scans: {total_scans}")
    print(f"Average distance between consecutive scans: {average_spacing:.4f}m")
    print(f"Median distance between consecutive scans: {median_spacing:.4f}m")
    print(f"Total path length: {total_path_length:.2f}m")
    print(f"Straight-line distance from first scan to last scan: {start_to_end_distance:.2f}m")
    print(
        f"Approximate number of scans needed to travel 1.5m "
        f"(based on average spacing): {scans_needed_to_cover_1_5m:.1f} scans"
    )







if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    site_folders = get_site_folders(project_root)

    for site_name, site_folder in site_folders.items():
        analyze_scan_spacing(site_folder, site_name)

    print("\nScan spacing analysis complete for all sites.")
    
    
    
    
    



# Site: wytham_data
# Total scans: 625
# Average distance between consecutive scans: 1.1124m
# Median distance between consecutive scans: 1.1092m
# Total path length: 694.11m
# Straight-line distance from first scan to last scan: 3.20m
# Approximate number of scans needed to travel 1.5m (based on average spacing): 1.3 scans


# Site: stein-am-rhein_data
# Total scans: 463
# Average distance between consecutive scans: 1.2212m
# Median distance between consecutive scans: 1.2020m
# Total path length: 564.17m
# Straight-line distance from first scan to last scan: 97.36m
# Approximate number of scans needed to travel 1.5m (based on average spacing): 1.2 scans


# Site: forest-of-dean_data
# Total scans: 625
# Average distance between consecutive scans: 1.1429m
# Median distance between consecutive scans: 1.1367m
# Total path length: 713.15m
# Straight-line distance from first scan to last scan: 38.00m
# Approximate number of scans needed to travel 1.5m (based on average spacing): 1.3 scans


# Site: evo_data
# Total scans: 1002
# Average distance between consecutive scans: 1.4789m
# Median distance between consecutive scans: 1.4862m
# Total path length: 1480.33m
# Straight-line distance from first scan to last scan: 141.84m
# Approximate number of scans needed to travel 1.5m (based on average spacing): 1.0 scans