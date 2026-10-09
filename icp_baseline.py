"""

Iterative Closest Point (another baseline for recognizing loop closures within a forest)

ICP, short for Iterative Closest Point, takes two scans and tries to slide and
turn one of them until it sits on top of the other, then measures how well
the two fit. 


# 1. Voxel downsampling
# Partition each point cloud into voxels with side length voxel_size.
# Replace all points in each occupied voxel with their centroid (mean x, y, z).
# If voxel_size = 0.5 m, each voxel measures 0.5 x 0.5 x 0.5 m.
# This reduces computation while approximately preserving the scan geometry.

# 2. Estimate initial yaw using Scan Context
# Represent each scan as a polar grid of radial rings and angular sectors.
# Store the maximum point height in each cell according to the descriptor setup.
# With 60 sectors, each circular column shift represents a 6-degree yaw change.
# Evaluate all shifts and retain the one minimizing the descriptor cosine distance.

# 3. Construct the initial transformation
# Build a 3x3 rotation matrix from the estimated yaw angle.
# Insert it into a 4x4 homogeneous transformation matrix T_init.
# Use the configured initial translation; yaw estimation alone provides no translation.
# Supply T_init to ICP as the starting transformation from source A to target B.

# 4. Find nearest-neighbor correspondences
# Transform source points using the current estimate of T.
# For each transformed source point, find the nearest target point in Euclidean distance.
# Reject correspondences beyond max_correspondence_distance, such as 1.5 m.
# Correspondences need not be one-to-one: several source points may share a target.
# These geometric associations are provisional, not verified physical-object matches.

# 5. Estimate a rigid transformation update
# For point-to-point ICP, minimize the sum of squared distances between matched points.
# Estimate a rotation R and translation t; scaling and deformation are not allowed.
# The standard solution uses matched-point centroids and SVD of their cross-covariance.
# Compose the estimated update with the current transformation.
# Point-to-plane ICP uses a different objective based on target surface normals.

# 6. Iterate correspondence search and transformation updates
# Recompute nearest neighbors after each transformation update.
# Reapply the correspondence-distance cutoff because accepted matches can change.
# Estimate another rigid update using the newly accepted correspondences.
# Repeat to refine the alignment; ICP can settle at a local rather than global optimum.
# The Scan Context initialization helps by providing an estimated starting yaw.

# 7. Check convergence and iteration limits
# Use max_iteration = 50 in the configuration described here.
# Monitor fitness and inlier RMSE after successive alignment updates.
# Fitness is accepted correspondences divided by the number of source points.
# Inlier RMSE is sqrt(mean squared distance over accepted correspondences).
# Stop when both changes meet the configured convergence tolerances or the limit is reached.
# Convergence does not establish that the scans form a correct loop closure.

# 8. Extract the scan-pair score
Fitness is the fraction of the points in Scan A that found a nearby point in B. For example:
- A has 100 points.
- 90 have a match within my (e.g., 1.5 m cutoff)
- Fitness = 90 ÷ 100 = 0.90. So 0.90 means 90% of source points in A found matches in B
higher fitness score means more points overlap.

# 9. Assign the reference revisit label
# Retrieve the original recorded positions associated with both scan timestamps.
# Compute their Euclidean separation in the coordinate dimensions used by the protocol.
# Mark the pair positive if it satisfies the revisit-distance rules.
# In our case, position distance <= 1.5 m and scan-index separation >= 50.


# 10. Evaluate the complete set of scan pairs
# Apply the same initialization, registration, and scoring procedure to every test pair.
# Store each pair's identifier, score, reference label, and useful registration metrics.
# Keep pair selection and labels unchanged when comparing ICP parameter settings.
# The composition of the selected pairs determines what this evaluation represents.
# Pair classification evaluates separation of labels; it does not directly measure Recall@1.


# 11. Calculate ROC-AUC
ROC-AUC checks whether scans of the same place get higher ICP scores than scans of 
different places. For example, a true-revisit pair scoring 0.90 beats a different-place 
pair scoring 0.30. AUC compares every true-revisit pair with every different-place pair. 
AUC = 0.889 means the true-revisit pair gets the higher fitness score in about 88.9% of the 
comparisons in the csv file (where all the respective scans are stored).


So this means, that we can account for and check loop-closures within a forest through this 
way (non-learned and purely geometric). So ICP is another loop closure baseline, like the Scan Context Algorithm.






"""

import os
import time
import numpy as np
import pandas as pd
import open3d as o3d
import rosbags
import open3d

from scan_context_baseline import (
    get_site_folders,
    load_scan_points,
    measure_scan_geometry,
    build_scan_context,
    column_cosine_distance,
    find_labels_file,
    compute_roc_auc,
    compute_best_f1,
)


def prepare_scan(scan_name, voxel_folder, remembered_scans, num_rings, num_sectors,
                 max_radius, floor_height, icp_voxel_size):
    
    """
    This function gets everything we need from one scan, given only its
    name. It loads the scan, builds its Scan Context picture, which we use
    to find the starting rotation, and makes a thinner copy of the point
    cloud for ICP to work on. ICP has to match every point against its
    nearest neighbour many times over, so running it on the full scan would
    take far too long across thousands of pairs, while a thinner copy keeps
    the shape of the forest and runs much faster. Each scan appears in many
    pairs, so the result is remembered and reused instead of being rebuilt.
    """
    
    if scan_name not in remembered_scans:
        points = load_scan_points(os.path.join(voxel_folder, f"{scan_name}.pcd"))
        descriptor = build_scan_context(points, num_rings, num_sectors, max_radius, floor_height)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        thin_cloud = cloud.voxel_down_sample(icp_voxel_size)

        remembered_scans[scan_name] = {"descriptor": descriptor, "cloud": thin_cloud}

    return remembered_scans[scan_name]



def find_best_rotation_shift(descriptor_a, descriptor_b):
    
    """
    This function finds how many pie slices one Scan Context picture has to
    be turned to look most like the other, the same search Scan Context
    itself performs. It tries every possible turn, measures the difference
    each time, and returns the number of slices for the best one. We do not
    need the difference score here, only the turn, because the turn tells
    us roughly how the two scans are rotated relative to each other, which
    is exactly the starting guess ICP needs.
    """
    
    num_sectors = descriptor_a.shape[1]
    best_shift = 0
    best_distance = float("inf")

    for shift in range(num_sectors):
        shifted_b = np.roll(descriptor_b, shift, axis=1)
        distance = column_cosine_distance(descriptor_a, shifted_b)
        if distance < best_distance:
            best_distance = distance
            best_shift = shift

    return best_shift


def build_yaw_transform(shift, num_sectors):
    
    """
    This function turns a number of pie slices into a 4 by 4 transformation
    matrix, which is the standard way to describe a rotation and a shift in
    3D. Each slice is a fixed angle, so the turn is the number of slices
    times that angle, and it is a spin around the vertical axis because the
    sensor stays upright. The shift part is left at zero, since both scans
    are centred on their own sensor and the sensors of a true same-place
    pair are less than 1.5 m apart, which is a small enough gap for ICP to
    close on its own.
    """
    
    angle = shift * 2.0 * np.pi / num_sectors
    cosine = np.cos(angle)
    sine = np.sin(angle)

    transform = np.eye(4)
    transform[0, 0] = cosine
    transform[0, 1] = -sine
    transform[1, 0] = sine
    transform[1, 1] = cosine
    return transform


def run_icp_for_pair(source_cloud, target_cloud, starting_transform, max_correspondence_distance):
    
    """
    This function runs ICP on one pair of thinned scans and returns its
    fitness. Starting from the given rotation, ICP repeatedly pairs every
    point of the source scan with its nearest point in the target scan,
    ignoring any pair further apart than the maximum correspondence
    distance, then adjusts the source scan to bring those pairs closer
    together, and repeats up to 50 times. The fitness it returns is the
    fraction of source points that ended up with a partner in range. It is
    near 1 when the two scans sit nicely on top of each other and near 0
    when they do not, so a high value suggests the same place.
    """
    
    if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
        return 0.0


# We use the ICP Library  ---> icp_baseline.py calls Open3D's registration_icp, so we haven't written ICP ourselves. Our code only prepares the scans, picks the starting rotation, and reads the fitness score.
# The library handles point matching, calculating rotation and movement, repeating, and stopping internally.
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        starting_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return float(result.fitness)



def score_pairs_for_site(site_folder, voxel_size, num_rings, num_sectors, max_radius,
                         floor_height, icp_voxel_size, max_correspondence_distance):
    
    """
    This function goes through every labeled pair of a site and gives it an
    ICP score. For each pair it finds the starting rotation from the two
    Scan Context pictures, runs ICP from that start, and records one minus
    the fitness as the difference score, so that, like Scan Context, a low
    number means the scans look like the same place. It returns the true
    labels and these scores in the same order, ready for the AUC and F1
    measures.
    """
    
    voxel_folder = os.path.join(site_folder, f"individual_clouds_voxel_{voxel_size:.2f}m")
    pairs_table = pd.read_csv(find_labels_file(site_folder))

    remembered_scans = {}
    labels = []
    distances = []
    total_pairs = len(pairs_table)

    for pair_number, row in enumerate(pairs_table.itertuples(index=False), start=1):
        scan_1 = prepare_scan(
            row.scan_1, voxel_folder, remembered_scans, num_rings, num_sectors,
            max_radius, floor_height, icp_voxel_size,
        )
        scan_2 = prepare_scan(
            row.scan_2, voxel_folder, remembered_scans, num_rings, num_sectors,
            max_radius, floor_height, icp_voxel_size,
        )

        shift = find_best_rotation_shift(scan_1["descriptor"], scan_2["descriptor"])
        starting_transform = build_yaw_transform(shift, num_sectors)

        fitness = run_icp_for_pair(
            scan_2["cloud"], scan_1["cloud"], starting_transform, max_correspondence_distance
        )

        distances.append(1.0 - fitness)
        labels.append(int(row.label))

        if pair_number % 250 == 0 or pair_number == total_pairs:
            print(f"    scored {pair_number}/{total_pairs} pairs")

    return np.array(labels), np.array(distances)




if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_size = 0.10
    num_rings = 20   # I thought to keep it consistent as Scan Context, for consistency (see scan_context_baseline.py)
    num_sectors = 60   # I thought to keep it consistent as Scan Context, for consistency (see scan_context_baseline.py)
    scans_per_site_for_geometry = 20
    icp_voxel_size = 0.5
    
    
    
    # This value shows the diagonal length of the cube, because within a cube, 
    # this is the max distance, the scans (pre-voxel downsampling might have existed)
    diagonal_length = round(icp_voxel_size * (3 ** 0.5), 2) # 0.87 in our case, as the cube (voxel size) is 0.5m from every dimension.
    
    
    
    run_sensitivity_test_first = True
    sensitivity_distances = [0.5, 0.75, diagonal_length, 1.5] # 0.87 is the length of the diagonal of the 0.5 m sided cube...
    chosen_max_correspondence_distance = 0.75

    site_folders = get_site_folders(project_root)

    print("Measuring scan geometry from a sample of scans:")
    max_radius, floor_height = measure_scan_geometry(
        site_folders, voxel_size, scans_per_site_for_geometry
    )
    

    if run_sensitivity_test_first:
        for site_name, site_folder in site_folders.items():
            print(f"\nSensitivity test on {site_name} (no final results yet)")
            for candidate_distance in sensitivity_distances:
                start_time = time.time()
                labels, distances = score_pairs_for_site(
                    site_folder, voxel_size, num_rings, num_sectors,
                    max_radius, floor_height, icp_voxel_size, candidate_distance,
                )
                auc = compute_roc_auc(labels, distances)
                elapsed = time.time() - start_time
                print(f"  max correspondence distance {candidate_distance} m: AUC {auc:.3f} ({elapsed:.0f} s)")

        print(
            "\nPick the best distance above, set chosen_max_correspondence_distance "
            "to it, set run_sensitivity_test_first to False, and run again."
        )
        
        
        
    else:
        results = {}
        for site_name, site_folder in site_folders.items():
            print(f"\nScoring pairs for {site_name}")
            try:
                labels, distances = score_pairs_for_site(
                    site_folder, voxel_size, num_rings, num_sectors, max_radius,
                    floor_height, icp_voxel_size, chosen_max_correspondence_distance,
                )
            except (FileNotFoundError, OSError) as error:
                print(f"  Skipping {site_name}: {error}")
                continue

            auc = compute_roc_auc(labels, distances)
            f1, precision, recall = compute_best_f1(labels, distances)
            results[site_name] = (auc, f1, precision, recall)


        print("\n \n")
        print("ICP baseline results")
        print(
            f"Settings: ICP voxel {icp_voxel_size} m, max correspondence distance "
            f"{chosen_max_correspondence_distance} m, start rotation from Scan Context"
        )
        print("\n \n")
        for site_name, (auc, f1, precision, recall) in results.items():
            print(
                f"{site_name:22s} AUC {auc:.3f} | best F1 {f1:.3f} "
                f"(precision {precision:.3f}, recall {recall:.3f})"
            )
