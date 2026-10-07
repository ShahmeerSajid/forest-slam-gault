

"""
SCAN CONTEXT BASELINE — OXFORD FOREST DATASET

WHAT IS SCAN CONTEXT AND WHY DO WE NEED IT?

Scan Context is a classical LiDAR place-recognition method. It converts a 3D
LiDAR scan into a compact fingerprint and compares fingerprints to determine
whether two scans were taken at the same place. We use it as a baseline:
before evaluating a learned model such as Logg3DNet, we need a simple,
established method to compare against. If the learned model performs better
than Scan Context, we can show the benefit of learning-based place recognition.


1. COMMON SCAN GEOMETRY
Before comparing scans, the code needs to decide:
How far away from the LiDAR sensor should we consider points?

It doesn't want Scan A to use 30 m, Scan B to use 70 m, etc. The fingerprints should cover
the same-sized area.
We sample 20 evenly spaced PCD scans from each of the four Oxford sites. Compute
each LiDAR point's horizontal distance sqrt(x² + y²) from the sensor. The 99th
percentile gives one common max radius (~51 m) for all scans.


2. SCAN CONTEXT FINGERPRINT
Suppose our scan pair is: Scan 10 + Scan 300
For each scan in a loop-closure pair, we divide the area within the common radius
into:
20 rings - divisions going outward from the sensor 
60 sectors - pizza-like slices going around the sensor

Therefore, 20 x 60 = 1,200 cells. Each cell stores its maximum point
height, producing a 20x60 fingerprint.

There could be many LiDAR points inside one cell with different heights. The code keeps
the maximum height point(s). It does this for all the 1200 cells. Eventually, Scan 10 becomes 
20 x 60 table of height values. 

This table is the Scan Context Descriptor/fingerprint for scan 10. 
So, one HUGE point cloud -> compact 20 x 60 fingerprint
We do the same for Scan 300.



3. The 2 Fingerprints' comparisons
The code compares their sectors.
One sector contains the values from the 20 rings.
It uses cosine distance to measure how similar the height patterns are.  

The important idea is simply:
Small distance = fingerprints look similar
Large distance = fingerprints look different

But there is a problem.
Imagine Scan 10 and Scan 300 were actually collected at the same physical place.
But:
Scan 10 → robot facing North ↑
Scan 300 → robot facing East →

The same trees and objects will appear in different sectors of the fingerprints.
So simply comparing the two fingerprints directly could make the same place appear different.

The code solves this by shifting/rotating Fingerprint B.
Because there are 60 sectors, it tries all 60 possible sector shifts.


4. LOOP-CLOSURE EVALUATION
Repeat for every pair in the loop-closure CSV. The CSV provides ground truth:
1 = same place, 0 = different place. These labels are used only for evaluation,
not for creating fingerprints or calculating Scan Context distances.


5. METRICS
AUC: "Does Scan Context generally give same-place pairs lower scores than different-place pairs?"
F1: "After choosing the best boundary, how good are the actual same/different decisions?"

For F1, the code automatically sorts the Scan Context distances and tries the possible decision boundaries.



The entire Scan Context pipeline:
Take two LiDAR scans from each of the 4 loop_closure csv files → turn each scan into a 20x60 height 
fingerprint using the same 51 m radius → rotate and compare the fingerprints → 
get a difference score → repeat for all labeled pairs → use AUC and F1 to measure 
how well those scores identify loop closures.

"""





import os
import numpy as np
import pandas as pd
import open3d as o3d
import rosbags



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








def load_scan_points(file_path):
    
    """
    This function loads one point cloud file from disk and returns just its
    x, y, and z coordinates as a plain NumPy array with one row per point.
    Scan Context only needs the raw coordinates, so we drop everything else
    and keep the rest of the code simple.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    return np.asarray(point_cloud.points, dtype=np.float32)






def pick_evenly_spaced_scan_files(voxel_folder, how_many):
    """
    Before comparing scans, the code needs to decide:
    How far away from the LiDAR sensor should we consider points?

    It doesn't want Scan A to use 30 m, Scan B to use 70 m, etc. The fingerprints
    should cover the same-sized area.
    
    So later, the code selects 20 evenly spaced scans from each of the four forests.
    
    """
    
    all_files = sorted(f for f in os.listdir(voxel_folder) if f.endswith(".pcd"))
    if len(all_files) <= how_many:
        return all_files
    chosen_positions = np.linspace(0, len(all_files) - 1, how_many).astype(int)
    return [all_files[position] for position in chosen_positions]







def measure_scan_geometry(site_folders, voxel_size, scans_per_site):
    """
    For every Lidar point in the sampled scans, we calcualte the horizontal distance
    from the sensor. It combines these distances and finds the 99th percentile. 
    Our result was 50.96 metres. So this one value is used across all the scans.
    """
    
    all_radii = []
    all_heights = []
    voxel_folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"

    for site_name, site_folder in site_folders.items():
        voxel_folder = os.path.join(site_folder, voxel_folder_name)
        if not os.path.isdir(voxel_folder):
            continue

        centre_distances = []
        for file_name in pick_evenly_spaced_scan_files(voxel_folder, scans_per_site):
            points = load_scan_points(os.path.join(voxel_folder, file_name))
            if len(points) == 0:
                continue
            all_radii.append(np.hypot(points[:, 0], points[:, 1]))
            all_heights.append(points[:, 2])
            centre_distances.append(np.linalg.norm(np.median(points[:, :2], axis=0)))

        if centre_distances:
            print(
                f"  {site_name}: median distance of scan centre from origin = "
                f"{np.median(centre_distances):.2f} m"
            )


    pooled_radii = np.concatenate(all_radii)
    pooled_heights = np.concatenate(all_heights)

    max_radius = float(np.percentile(pooled_radii, 99))
    floor_height = float(np.percentile(pooled_heights, 1))


    print(f"  99th percentile of distance from sensor: {max_radius:.2f} m")
    print(f"  1st percentile of height: {floor_height:.2f} m")
    return max_radius, floor_height









def build_scan_context(points, num_rings, num_sectors, max_radius, floor_height):
    
    """
    This function turns one scan into its Scan Context picture. Looking
    down from above with the sensor at the centre, it cuts the surroundings
    into rings, like the rings of a target, and into sectors, like slices
    of a pie. That gives a grid of cells, num_rings tall and num_sectors
    wide. For every cell we store the height of the tallest point inside
    it, so a cell with a tree trunk or branch in it holds a large number
    and an empty cell holds zero. 
    
    The result is a small grid of numbers that summarizes the shape of the 
    forest around the sensor, which is what we compare between scans. 
    Heights are shifted by the floor height so that the ground is near zero, 
    and points beyond the maximum radius are ignored.
    
    """
    
    descriptor = np.zeros((num_rings, num_sectors), dtype=np.float32)
    if len(points) == 0:
        return descriptor

    radius = np.hypot(points[:, 0], points[:, 1])
    angle = np.arctan2(points[:, 1], points[:, 0])
    height = np.maximum(points[:, 2] - floor_height, 0.0)

    inside = radius < max_radius
    radius = radius[inside]
    angle = angle[inside]
    height = height[inside]

    ring_index = np.minimum((radius / max_radius * num_rings).astype(int), num_rings - 1)
    sector_index = np.minimum(
        ((angle + np.pi) / (2 * np.pi) * num_sectors).astype(int), num_sectors - 1
    )

    np.maximum.at(descriptor, (ring_index, sector_index), height)
    return descriptor






def column_cosine_distance(descriptor_a, descriptor_b):
    
    """
    This function measures how different two Scan Context pictures are,
    assuming they are already lined up. It compares them one column at a
    time, where a column is one pie slice seen from the centre outward. For
    each pair of columns it checks whether they point in the same
    direction, which is a way of asking whether the heights rise and fall
    in the same pattern going outward. It then averages this over all
    columns that actually contain something in both pictures. The answer
    is a number from 0 to 1, where 0 means the pictures look identical and
    larger numbers mean they look more different.
    
    """
    norms_a = np.linalg.norm(descriptor_a, axis=0)
    norms_b = np.linalg.norm(descriptor_b, axis=0)
    usable = (norms_a > 0) & (norms_b > 0)

    if not usable.any():
        return 1.0

    dot_products = np.sum(descriptor_a * descriptor_b, axis=0)
    cosine_similarity = dot_products[usable] / (norms_a[usable] * norms_b[usable])
    return float(np.mean(1.0 - cosine_similarity))








def scan_context_distance(descriptor_a, descriptor_b):
    
    """
    This function gives the final difference score between two scans. The
    catch is that the same place looks different if the sensor is facing a
    different direction when the scan is taken, which would turn the whole
    picture sideways. To handle this, we slide one picture around its
    sectors one step at a time, as if turning the sensor, measure the
    difference at every turn, and keep the best, meaning the smallest,
    score. A low final score means there is some rotation under which the
    two scans look alike, so they are probably the same place.
    """
    
    num_sectors = descriptor_a.shape[1]
    best_distance = 1.0

    for shift in range(num_sectors):
        shifted_b = np.roll(descriptor_b, shift, axis=1)
        distance = column_cosine_distance(descriptor_a, shifted_b)
        if distance < best_distance:
            best_distance = distance

    return best_distance





def find_labels_file(site_folder):
    """
    This function finds a site's labeled pairs file, which tells us which
    scan pairs are the same place and which are not. It checks for the
    site-specific filename first and the plain filename second, the same
    way our Dataset code does.
    """
    site_name = os.path.basename(os.path.normpath(site_folder))
    for name in [f"{site_name}_loop_closure_pairs.csv", "loop_closure_pairs.csv"]:
        candidate_path = os.path.join(site_folder, name)
        if os.path.isfile(candidate_path):
            return candidate_path
    raise FileNotFoundError(f"No loop_closure_pairs.csv file found in {site_folder}")






def get_or_build_descriptor(scan_name, voxel_folder, remembered_descriptors, num_rings,
                            num_sectors, max_radius, floor_height):
    """
    This function returns the Scan Context picture for one scan, given just
    its name. If we have built it before, it hands back the remembered copy
    straight away. If not, it loads the scan from disk, builds the picture,
    remembers it, and returns it. We do this because the same scan appears
    in many different pairs, and loading and rebuilding it every time would
    make the whole run far slower for no benefit.
    """
    if scan_name not in remembered_descriptors:
        points = load_scan_points(os.path.join(voxel_folder, f"{scan_name}.pcd"))
        remembered_descriptors[scan_name] = build_scan_context(
            points, num_rings, num_sectors, max_radius, floor_height
        )
    return remembered_descriptors[scan_name]





def score_all_pairs_for_site(site_folder, voxel_size, num_rings, num_sectors, max_radius, floor_height):
    """
    This function goes through every labeled pair of a site, builds the
    Scan Context picture for both scans, and records the difference score
    between them next to the true label. Many pairs share the same scans,
    so each scan's picture is built only once and remembered, which saves a
    lot of time. It returns the true labels and the scores as two lists, in
    the same order, which is exactly what we need to measure how well the
    scores separate real matches from non-matches.
    """
    voxel_folder = os.path.join(site_folder, f"individual_clouds_voxel_{voxel_size:.2f}m")
    pairs_table = pd.read_csv(find_labels_file(site_folder))

    remembered_descriptors = {}
    labels = []
    distances = []
    total_pairs = len(pairs_table)

    for pair_number, row in enumerate(pairs_table.itertuples(index=False), start=1):
        descriptor_1 = get_or_build_descriptor(
            row.scan_1, voxel_folder, remembered_descriptors,
            num_rings, num_sectors, max_radius, floor_height,
        )
        descriptor_2 = get_or_build_descriptor(
            row.scan_2, voxel_folder, remembered_descriptors,
            num_rings, num_sectors, max_radius, floor_height,
        )
        distances.append(scan_context_distance(descriptor_1, descriptor_2))
        labels.append(int(row.label))

        if pair_number % 500 == 0 or pair_number == total_pairs:
            print(f"    scored {pair_number}/{total_pairs} pairs")

    return np.array(labels), np.array(distances)






def compute_roc_auc(labels, distances):
    """
    This function computes the ROC AUC, a single number between 0 and 1
    that answers one question: if we pick one true same-place pair and one
    different-place pair at random, how often does the same-place pair get
    the lower difference score? A value of 0.5 means the scores are no
    better than guessing, and 1.0 means they separate the two kinds of
    pairs perfectly. A useful property is that this number does not change
    with how many negative pairs we chose to include, so it is safe to
    compare across sites.
    """
    positive_distances = distances[labels == 1]
    negative_distances = distances[labels == 0]

    if len(positive_distances) == 0 or len(negative_distances) == 0:
        return float("nan")

    wins = np.mean(positive_distances[:, None] < negative_distances[None, :])
    ties = np.mean(positive_distances[:, None] == negative_distances[None, :])
    return float(wins + 0.5 * ties)





def compute_best_f1(labels, distances):
    """
    This function finds the best F1 score we could reach by choosing the
    ideal cut-off. We sort all pairs from the lowest difference score to
    the highest, then imagine drawing a line at every possible position,
    calling everything before the line a match. At each position we
    measure precision, which is how many of the claimed matches are real,
    and recall, which is how many of the real matches we found, and combine
    them into F1. We return the best one with its precision and recall.
    Unlike ROC AUC, this number depends on our choice of three negative
    pairs for every positive pair, so it should only be compared between
    methods that use the very same pairs.
    """
    
    order = np.argsort(distances)
    sorted_labels = labels[order]

    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)
    total_positives = sorted_labels.sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_positives
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    best_index = int(np.argmax(f1_scores))
    return float(f1_scores[best_index]), float(precision[best_index]), float(recall[best_index])





if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_size = 0.10
    num_rings = 20
    num_sectors = 60
    scans_per_site_for_geometry = 20

    site_folders = get_site_folders(project_root)

    print("Measuring scan geometry from a sample of scans:")
    max_radius, floor_height = measure_scan_geometry(
        site_folders, voxel_size, scans_per_site_for_geometry
    )

    results = {}
    for site_name, site_folder in site_folders.items():
        print(f"\nScoring pairs for {site_name}")
        try:
            labels, distances = score_all_pairs_for_site(
                site_folder, voxel_size, num_rings, num_sectors, max_radius, floor_height
            )
        except (FileNotFoundError, OSError) as error:
            print(f"  Skipping {site_name}: {error}")
            continue

        auc = compute_roc_auc(labels, distances)
        f1, precision, recall = compute_best_f1(labels, distances)
        results[site_name] = (auc, f1, precision, recall)

    print("\n \n")
    print("Scan Context baseline results")
    print(f"Settings: {num_rings} rings, {num_sectors} sectors, "
          f"max radius {max_radius:.1f} m, voxel size {voxel_size} m")
    print("==" * 30)
    for site_name, (auc, f1, precision, recall) in results.items():
        print(
            f"{site_name:22s} AUC {auc:.3f} | best F1 {f1:.3f} "
            f"(precision {precision:.3f}, recall {recall:.3f})"
        )
