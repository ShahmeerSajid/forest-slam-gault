
"""
Positive Pair Verification Script

Our labeling script reported how many same-place scan pairs each Oxford
site has, and some of those counts are quite small. Before I accept them,
I wanted another script that recounts the pairs on its own, using a different 
and faster way and reports extra facts about each site's path. 

Both scripts use the same loop-closure rule: two scans are a positive pair if they 
are within 1.5 m and at least 50 scans apart. The original labeling script 
checks later scans one by one using Python loops, while this new script
uses NumPy to check all later scans at once, so it is faster; if both produce
the same count, it helps confirm the labeling is correct. 


One of the research papers sent by William had vertical drift issue, so we wanted to account 
for this issue too:
This code also checks 2D (x,y) distance to see whether height/vertical drift affects the labels, 
reports how many unique scans participate in loop closures, gives basic trajectory
information such as path length, start-to-end distance, and height range, and confirms 
the scans are ordered by time.

"""

import os
import numpy as np
import pandas as pd
import rosbags



def get_site_folders(project_root):

    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
    site_paths = {}
    for site_name in site_names:
        site_paths[site_name] = os.path.join(project_root, site_name)
    return site_paths


def count_close_pairs(coordinates, distance_threshold, min_sequence_gap):
    
    """
    This function counts scan pairs that are close enough in distance and at least 50 scans apart.
    For each scan, it checks all eligible later scans at once, making it faster.
    It also counts how many different scans are involved in these pairs.
    One revisit can create several nearby scan pairs, so number of pairs != number of revisits.

    """
    
    total_scans = len(coordinates)
    pair_count = 0
    scans_involved = set()

    for i in range(total_scans - min_sequence_gap):
        later_coordinates = coordinates[i + min_sequence_gap:]
        distances = np.linalg.norm(later_coordinates - coordinates[i], axis=1)
        close_positions = np.nonzero(distances <= distance_threshold)[0]

        if len(close_positions) > 0:
            pair_count += len(close_positions)
            scans_involved.add(i)
            for position in close_positions:
                scans_involved.add(i + min_sequence_gap + int(position))

    return pair_count, len(scans_involved)




def describe_site(site_folder, site_name, distance_threshold, min_sequence_gap):

    """
    This function loads one site's slam_poses.csv and checks why the site has that number 
    of loop-closure pairs. 
    It shows the number of scans, path length, start-to-end distance, height range, and time order.
    It also compares 3D pair counts (x, y, z) with flat 2D counts (x, y).
    If the 2D count is much higher, height differences may be hiding real revisits.
    
    
    """
    poses_path = os.path.join(site_folder, "slam_poses.csv")
    if not os.path.isfile(poses_path):
        print(f"Skipping {site_name}: no slam_poses.csv found")
        return

    poses_table = pd.read_csv(poses_path)
    positions = poses_table[["x", "y", "z"]].values.astype(np.float64)

    step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    print(f"\nSite: {site_name}")
    print(f"  Scans: {len(positions)}")
    print(f"  Total path walked: {step_lengths.sum():.1f} m")
    print(f"  Start to end straight-line distance: {np.linalg.norm(positions[-1] - positions[0]):.1f} m")
    print(f"  Height range of path (max z minus min z): {positions[:, 2].max() - positions[:, 2].min():.1f} m")

    if "timestamp" in poses_table.columns:
        in_order = bool(np.all(np.diff(poses_table["timestamp"].values) >= 0))
        print(f"  Rows in time order: {in_order}")

    pairs_3d, scans_3d = count_close_pairs(positions, distance_threshold, min_sequence_gap)
    pairs_flat, scans_flat = count_close_pairs(positions[:, :2], distance_threshold, min_sequence_gap)

    print(f"  Positive pairs using 3D distance (same as labeling script): {pairs_3d}, involving {scans_3d} scans")
    print(f"  Positive pairs using flat x,y distance only: {pairs_flat}, involving {scans_flat} scans")







if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    distance_threshold_meters = 1.5
    min_sequence_gap_scans = 50

    site_folders = get_site_folders(project_root)

    for site_name, site_folder in site_folders.items():
        describe_site(site_folder, site_name, distance_threshold_meters, min_sequence_gap_scans)

    print("\nVerification complete.")





# Results:

# Site: wytham_data
#   Scans: 625
#   Total path walked: 694.1 m
#   Start to end straight-line distance: 3.2 m
#   Height range of path (max z minus min z): 10.6 m
#   Rows in time order: True
#   Positive pairs using 3D distance (same as labeling script): 60, involving 57 scans
#   Positive pairs using flat x,y distance only: 62, involving 57 scans

# Site: stein-am-rhein_data
#   Scans: 463
#   Total path walked: 564.2 m
#   Start to end straight-line distance: 97.4 m
#   Height range of path (max z minus min z): 2.6 m
#   Rows in time order: True
#   Positive pairs using 3D distance (same as labeling script): 394, involving 249 scans
#   Positive pairs using flat x,y distance only: 395, involving 250 scans

# Site: forest-of-dean_data
#   Scans: 625
#   Total path walked: 713.2 m
#   Start to end straight-line distance: 38.0 m
#   Height range of path (max z minus min z): 7.2 m
#   Rows in time order: True
#   Positive pairs using 3D distance (same as labeling script): 71, involving 68 scans
#   Positive pairs using flat x,y distance only: 71, involving 68 scans

# Site: evo_data
#   Scans: 1002
#   Total path walked: 1480.3 m
#   Start to end straight-line distance: 141.8 m
#   Height range of path (max z minus min z): 4.4 m
#   Rows in time order: True
#   Positive pairs using 3D distance (same as labeling script): 1098, involving 572 scans
#   Positive pairs using flat x,y distance only: 1102, involving 572 scans


# According to the results, the 3D and flat 2D loop-closure counts are almost identical, 
# showing that height differences are not hiding significant revisits. 
# Wytham and Forest of Dean naturally have fewer loop 
# closures because fewer scans participate in revisits, while Stein-am-Rhein and especially 
# Evo contain many more revisited areas. Overall, the differences in positive-pair counts
# appear to come from the different paths taken through each forest, rather than a problem 
# with the labeling method. 