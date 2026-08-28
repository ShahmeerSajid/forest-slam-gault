

# The core problem: we need a rule that separates these two
# situations: "genuinely came back later" vs. "just still walking through the same spot."
# EXAMPLE:


# Imagine the robot walks past a tree, scan #10. Then it keeps walking, and 40 scans 
# later, at scan #50, it happens to walk past that same tree again from a different 
# direction. Are scan #10 and scan #50 a "real revisit"? We'd say yes as 40 scans is 
# a lot of walking in between (roughly 40-60 meters), so this is a genuine return to
# that spot after wandering elsewhere.
# Now imagine instead: scan #10 and scan #15 are close in position. Is that a 
# "real revisit"? Almost certainly no because the robot just took 5 scans while slowly 
# walking through that one area. It didn't leave and come back; it's simply still there.




# The purpose of this code is to figure out "which scan pairs are actually the same place" ?
# The code looks at every pair of scans in a site and applies two rules together: 

#     1) Are they physically close (within 1.5m)? : checked using their x,y,z positions.
#     2) Are they far apart in the sequence (at least some number of scans apart,
#     so it's not just consecutive movement)? — this is the "gap" we were testing.

# If both these rules fall true, that pair gets labeled a positive
# (same place, different time makes it a real revisit). We also grab some random far-apart
# pairs as negatives (different places). All of this gets saved into a CSV per 
# site. THIS CSV FILE WILL BE MY TRAINING DATA NOW.





import pandas as pd
import numpy as np
import os
import random
import open3d as o3d







def get_site_folders(project_root):
    """
    This function defines the four Oxford forest sites we are working with
    and builds the full folder path for each one, based on where the
    project lives on disk. Keeping this list in one place means that if a
    folder name ever changes, we only need to update it here.
    """
    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
    site_paths = {}
    for site_name in site_names:
        site_paths[site_name] = os.path.join(project_root, site_name)
    return site_paths






def load_poses(site_folder):
    """
    This function loads the slam_poses.csv file for a given site into a
    pandas table, which lets us work with the scan positions easily. Each
    row in this file represents one scan, and tells us its file name, the
    time it was recorded, its position in 3D space as x, y, and z
    coordinates, and its orientation as a quaternion. For our loop-closure
    labeling, we mainly care about the scan name and its x, y, z position,
    since we are identifying revisits based on how physically close two
    scans are to each other.
    """
    poses_path = os.path.join(site_folder, "slam_poses.csv")

    if not os.path.isfile(poses_path):
        return None

    poses_table = pd.read_csv(poses_path)
    return poses_table







def find_positive_pairs(poses_table, distance_threshold, min_sequence_gap):
    """
    This function goes through every possible pair of scans in the site and
    checks two things: first, whether the two scans are physically close to
    each other, using straight-line distance between their x, y, z
    positions, and second, whether the two scans are far apart in the
    sequence they were recorded in. We require both conditions together
    because two scans that are simply next to each other in time will
    almost always be physically close too, which is not a meaningful
    revisit, just normal continuous movement. Only when a pair is close in
    position but far apart in time do we consider it real evidence that the
    robot has returned to a place it already visited, so we label that pair
    as a positive loop-closure pair.
    """
    positions = poses_table[["x", "y", "z"]].values
    scan_names = poses_table["individual_cloud"].values
    total_scans = len(poses_table)

    positive_pairs = []

    for i in range(total_scans):
        for j in range(i + min_sequence_gap, total_scans):
            distance = np.linalg.norm(positions[i] - positions[j])

            if distance <= distance_threshold:
                positive_pairs.append((scan_names[i], scan_names[j], distance))

    return positive_pairs









def sample_negative_pairs(poses_table, distance_threshold, num_negatives_needed):
    """
    This function randomly samples pairs of scans that are far apart in
    position, to use as negative examples, meaning pairs that are
    genuinely not the same place. Rather than checking every single
    possible far-apart pair, which would be an enormous and unnecessary
    number, we randomly pick candidate pairs and keep the ones that turn
    out to be far enough apart, stopping once we have collected enough
    negative examples to reasonably balance against our positive pairs.
    """
    positions = poses_table[["x", "y", "z"]].values
    scan_names = poses_table["individual_cloud"].values
    total_scans = len(poses_table)

    negative_pairs = []
    attempts = 0
    max_attempts = num_negatives_needed * 20

    while len(negative_pairs) < num_negatives_needed and attempts < max_attempts:
        i = random.randint(0, total_scans - 1)
        j = random.randint(0, total_scans - 1)
        attempts += 1

        if i == j:
            continue

        distance = np.linalg.norm(positions[i] - positions[j])

        if distance > distance_threshold:
            negative_pairs.append((scan_names[i], scan_names[j], distance))

    return negative_pairs








def test_sequence_gap_values(poses_table, distance_threshold, gap_values_to_test):
    """
    This function helps us pick a defensible minimum sequence gap instead
    of guessing one number. Rather than committing to a single gap value,
    we test several candidate values, such as 30, 50, and 100 scans apart,
    and for each one, count how many positive loop-closure pairs would
    actually be found in this site's data. A gap that is too small will
    produce a lot of positive pairs, but some of them might just be the
    robot's path briefly curving near itself rather than a genuine
    separate return visit. A gap that is too large will be more strict and
    trustworthy, but may produce very few positive pairs, which risks
    leaving us without enough real examples to work with. Seeing the actual
    counts side by side lets us make this choice based on evidence from our
    own data, rather than a guess.
    """
    positions = poses_table[["x", "y", "z"]].values
    total_scans = len(poses_table)

    results = {}

    for gap_value in gap_values_to_test:
        count = 0
        for i in range(total_scans):
            for j in range(i + gap_value, total_scans):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= distance_threshold:
                    count += 1
        results[gap_value] = count

    return results





def build_labels_for_site(site_folder, site_name, distance_threshold, min_sequence_gap, negative_ratio):
    """
    This function ties everything together for one site. It loads the pose
    data, finds all true positive loop-closure pairs, then samples a
    proportional number of negative pairs based on how many positives we
    found, and finally saves both sets combined into a single CSV file with
    a label column, where 1 means the pair is a true loop closure and 0
    means it is not. This output file becomes the ground truth our model
    will later be trained and evaluated against.
    """
    
    poses_table = load_poses(site_folder)

    if poses_table is None:
        print(f"Skipping {site_name}: no slam_poses.csv found")
        return

    positive_pairs = find_positive_pairs(poses_table, distance_threshold, min_sequence_gap)
    num_positives = len(positive_pairs)

    if num_positives == 0:
        print(f"Warning for {site_name}: no positive pairs found with current thresholds")
        return

    num_negatives_needed = num_positives * negative_ratio
    negative_pairs = sample_negative_pairs(poses_table, distance_threshold, num_negatives_needed)

    rows = []
    for scan_1, scan_2, distance in positive_pairs:
        rows.append({"scan_1": scan_1, "scan_2": scan_2, "distance": distance, "label": 1})

    for scan_1, scan_2, distance in negative_pairs:
        rows.append({"scan_1": scan_1, "scan_2": scan_2, "distance": distance, "label": 0})

    labels_table = pd.DataFrame(rows)
    output_path = os.path.join(site_folder, "loop_closure_pairs.csv")
    labels_table.to_csv(output_path, index=False)

    print(
        f"{site_name}: {num_positives} positive pairs, "
        f"{len(negative_pairs)} negative pairs, saved to {output_path}"
    )








if (__name__ == "__main__"):
    
    project_root = os.path.dirname(os.path.abspath(__file__))

    distance_threshold_meters = 1.5
    negative_to_positive_ratio = 3

    site_folders = get_site_folders(project_root)

    run_gap_test_first = True



    if (run_gap_test_first):
        gap_values_to_test = [30, 50, 75, 100]

        print("Testing different minimum sequence gap values (no files saved yet)...")

        for site_name, site_folder in site_folders.items():
            poses_table = load_poses(site_folder)

            if poses_table is None:
                print(f"Skipping {site_name}: no slam_poses.csv found")
                continue

            print(f"\nSite: {site_name}")
            gap_results = test_sequence_gap_values(
                poses_table, distance_threshold_meters, gap_values_to_test
            )

            for gap_value, count in gap_results.items():
                print(f"  Minimum gap {gap_value} scans: {count} positive pairs found")

        print(
            "\nReview the counts above, choose a minimum sequence gap, then set "
            "run_gap_test_first to False and set min_sequence_gap_scans below "
            "to generate the actual labeled files."
        )
        
        
        





    else:
        min_sequence_gap_scans = 50

        for site_name, site_folder in site_folders.items():
            build_labels_for_site(
                site_folder,
                site_name,
                distance_threshold_meters,
                min_sequence_gap_scans,
                negative_to_positive_ratio,
            )




        print("\nGround truth labeling complete for all sites.")
        
        
        
        
        
        
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
      
        
