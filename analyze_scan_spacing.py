

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
    
    
    
    
    

# Conclusion I drew from the results:

# Site: wytham_data
#   Minimum gap 30 scans: 60 positive pairs found
#   Minimum gap 50 scans: 60 positive pairs found
#   Minimum gap 75 scans: 54 positive pairs found
#   Minimum gap 100 scans: 51 positive pairs found

# Site: stein-am-rhein_data
#   Minimum gap 30 scans: 417 positive pairs found
#   Minimum gap 50 scans: 394 positive pairs found
#   Minimum gap 75 scans: 331 positive pairs found
#   Minimum gap 100 scans: 305 positive pairs found

# Site: forest-of-dean_data
#   Minimum gap 30 scans: 113 positive pairs found
#   Minimum gap 50 scans: 71 positive pairs found
#   Minimum gap 75 scans: 67 positive pairs found
#   Minimum gap 100 scans: 67 positive pairs found

# Site: evo_data
#   Minimum gap 30 scans: 1109 positive pairs found
#   Minimum gap 50 scans: 1098 positive pairs found
#   Minimum gap 75 scans: 1043 positive pairs found
#   Minimum gap 100 scans: 927 positive pairs found


# Each number: 30, 50, 75, and 100 scans is just a different rule for deciding how far apart two scans need to be in the robot's journey before we call it a real revisit.
# 30 scans: More relaxed → finds more possible loop closures, but some could be normal movement.
# 50 scans: A middle/balanced choice.
# 75 scans: More strict -> fewer pairs.
# 100 scans: Very strict -> fewer pairs, but the pairs are more likely to be real revisits.
# When we tested them, increasing the gap from 30 to 100 didn't remove a huge number of pairs. The number of positive pairs only decreased by about 10–25% at most sites.
# This suggests that most of the pairs we found at 30 scans were probably real revisits, rather than just the robot slowly moving through the same area. 
# So we chose 50 scans as a reasonable middle ground, strict enough to avoid obvious false positives, but not so strict that we lose too many real loop closures.