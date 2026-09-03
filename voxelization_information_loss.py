"""

Point count reduction alone does not tell us how much real detail we are
losing when we voxelize a scan, since it only measures data size, not
accuracy. This script measures actual geometric information loss instead,
by checking how far the voxelized version of a scan has drifted from the
original scan's true shape. This gives us a real, defensible number to
compare voxel sizes with, rather than relying on visual impressions alone.


For each scan, it checks: after voxelizing, how far (in meters) is each original point from 
its nearest point in the voxelized version? It averages that across the whole 
scan, giving one number per voxel size that reflects actual detail lost, not 
just how many points were removed. Lower number = less distortion = better preserved shape.

This information is telling us how much geometric detail we lose when we voxelize 
the LiDAR scans at different voxel sizes. We need it because choosing a voxel size 
is a trade-off: larger voxels reduce the amount of data and make processing 
faster, but they also distort the original point cloud more. 
From the results, the numbers show that 0.1 m causes only about 1 cm of average 
positional distortion, while 0.2 m causes about 4 cm and 0.4 m about 10 cm. 
So this gives me a measurable reason to choose 0.1 m rather than simply 
saying, “0.1 m looks good visually.” 

This is a metric rather than comparing the different voxel sizes through just asking "how much points were reduced."


"""



import open3d as o3d
import numpy as np
import os
import pandas as pd
import numpy as np





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







def compute_information_loss_for_scan(original_cloud, voxel_size):
    
    """
    This function measures how much geometric detail is lost when a single
    scan is voxelized at a given voxel size. It works by first creating the
    voxelized version of the scan, and then, for every point in the
    original scan, finding the distance to the closest point in the
    voxelized version. If voxelization preserved the scan's shape
    perfectly, these distances would all be very small, since every
    original point would still have a very close match nearby. If
    voxelization is throwing away real detail, these distances grow larger,
    because entire regions of the original shape are no longer represented
    closely by the voxelized version. We then average these distances
    across the whole scan to get one single number representing how much
    detail was lost for that scan, at that voxel size. A smaller number
    means less detail was lost, and a larger number means more detail was
    lost.
    """
    
    downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)

    distances = original_cloud.compute_point_cloud_distance(downsampled_cloud)
    distances = np.asarray(distances)

    average_distance = np.mean(distances)
    return average_distance







def compute_average_information_loss_for_site(site_folder, voxel_sizes):
    
    """
    This function goes through every scan in one site's individual_clouds
    folder, computes the information loss at each voxel size we care about
    using the function above, and keeps a running total for each voxel
    size. At the end, it divides each total by the number of scans to get
    the average information loss across the whole site, for each voxel
    size. This mirrors how we already computed average point reduction, but
    now measuring actual geometric accuracy instead of just data size.
    """
    
    individual_clouds_path = os.path.join(site_folder, "individual_clouds")

    if not os.path.isdir(individual_clouds_path):
        return None

    all_files = [f for f in os.listdir(individual_clouds_path) if f.endswith(".pcd")]
    total_files = len(all_files)

    if total_files == 0:
        return None

    loss_totals = {voxel_size: 0.0 for voxel_size in voxel_sizes}

    for file_name in all_files:
        file_path = os.path.join(individual_clouds_path, file_name)
        original_cloud = o3d.io.read_point_cloud(file_path)

        if len(original_cloud.points) == 0:
            continue

        for voxel_size in voxel_sizes:
            average_distance = compute_information_loss_for_scan(original_cloud, voxel_size)
            loss_totals[voxel_size] += average_distance

    average_losses = {
        voxel_size: (loss_totals[voxel_size] / total_files) for voxel_size in voxel_sizes
    }

    return average_losses, total_files






def print_summary_table(all_site_results, voxel_sizes):
    
    """
    This function prints out the average information loss results for
    every site as a simple, readable table, with one row per site and one
    column per voxel size, measured in meters. Since these numbers
    represent an average distance in meters between the original scan and
    its voxelized version, smaller numbers are better, and this table makes
    it easy to see whether that pattern holds consistently across all four
    forests, or whether some sites lose more detail than others at the same
    voxel size.
    """
    
    print("\nAverage Information Loss Summary (meters, lower is better)")
    print("\n")

    header = f"{'Site':<22}" + "".join(f"{f'{v}m':>14}" for v in voxel_sizes)
    print(header)
    print("\n")

    for site_name, (average_losses, total_files) in all_site_results.items():
        row = f"{site_name:<22}"
        for voxel_size in voxel_sizes:
            row += f"{average_losses[voxel_size]:>13.4f}m"
        print(row)
        print(f"  (based on {total_files} scans)")

    print("\n")





if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_sizes_to_test = [0.05, 0.10, 0.20, 0.40]

    site_folders = get_site_folders(project_root)
    all_site_results = {}

    for site_name, site_folder in site_folders.items():
        print(f"\nProcessing site: {site_name}")
        result = compute_average_information_loss_for_site(site_folder, voxel_sizes_to_test)

        if result is None:
            print(f"Skipping {site_name}: no individual_clouds folder or no .pcd files found")
            continue

        average_losses, total_files = result
        all_site_results[site_name] = (average_losses, total_files)

        for voxel_size in voxel_sizes_to_test:
            print(f"  Voxel size {voxel_size}m: {average_losses[voxel_size]:.4f}m average information loss")

    if all_site_results:
        print_summary_table(all_site_results, voxel_sizes_to_test)
    else:
        print("\nNo sites were processed. Check that individual_clouds folders exist and contain .pcd files.")
        
        
        
        
        
        
        
        


# Processing site: wytham_data
#   Voxel size 0.05m: 0.0022m average information loss
#   Voxel size 0.1m: 0.0100m average information loss
#   Voxel size 0.2m: 0.0356m average information loss
#   Voxel size 0.4m: 0.0942m average information loss

# Processing site: stein-am-rhein_data
#   Voxel size 0.05m: 0.0042m average information loss
#   Voxel size 0.1m: 0.0139m average information loss
#   Voxel size 0.2m: 0.0401m average information loss
#   Voxel size 0.4m: 0.1026m average information loss

# Processing site: forest-of-dean_data
#   Voxel size 0.05m: 0.0027m average information loss
#   Voxel size 0.1m: 0.0114m average information loss
#   Voxel size 0.2m: 0.0388m average information loss
#   Voxel size 0.4m: 0.1017m average information loss

# Processing site: evo_data
#   Voxel size 0.05m: 0.0031m average information loss
#   Voxel size 0.1m: 0.0114m average information loss
#   Voxel size 0.2m: 0.0348m average information loss
#   Voxel size 0.4m: 0.0895m average information loss




# Average Information Loss Summary (meters, lower is better)


# Site                           0.05m          0.1m          0.2m          0.4m


# wytham_data                  0.0022m       0.0100m       0.0356m       0.0942m
#   (based on 625 scans)
# stein-am-rhein_data          0.0042m       0.0139m       0.0401m       0.1026m
#   (based on 463 scans)
# forest-of-dean_data          0.0027m       0.0114m       0.0388m       0.1017m
#   (based on 625 scans)
# evo_data                     0.0031m       0.0114m       0.0348m       0.0895m
#   (based on 1002 scans)




# Conclusion: 0.1 m is the best overall compromise. It reduces the number of LiDAR 
# points by about 50–59%, which makes the data easier/faster to process, while the average 
# positional distortion stays around 1 cm. Larger voxels save more data but introduce much 
# more distortion—about 3.5–4 cm at 0.2 m and 9–10 cm at 0.4 m. The visual comparisons 
# also showed that 0.1 m keeps the important forest structure and trunks while mainly 
# simplifying very fine details. So, for the place-recognition work, I have decided to 
# choose 0.1m as the voxelization size. Therefore, Standardize on 0.1m going 
# forward — all future work (labeling, baselines, training) uses the 
# individual_clouds_voxel_0.10m/ folders instead of raw scans.
# So 0.1 m gives me a significant reduction in data without sacrificing too much of 
# the geometry that the algorithm needs to recognize places.



# So just to summarize, we're going with 0.1m becuse it seems like the best 
# balance: it reduces the point count by ~50–59% while keeping average positional 
# distortion to only ~1cm across all four sites. Compared to 0.2m (~4cm) and 
# 0.4m (~9–10cm), it preserves the geometry much better while still giving a 
# significant reduction in data.  


