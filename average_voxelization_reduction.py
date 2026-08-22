"""
Average Voxelization Reduction Script
Forest SLAM Project - Oxford Dataset Preprocessing

This script goes through every scan in every forest site and works out the
average point count reduction at each voxel size we have been testing
(0.05m, 0.10m, 0.20m). Instead of just looking at a handful of sample
scans, this gives us a full picture across the entire
dataset for each site, which is a more reliable basis for deciding which
voxel size to standardize on going forward.

---> I am doing this so I can get an idea of how much there is a reduction of data points
based on the 


"""








import open3d as o3d
import os
import pandas as pd
import numpy as npsss






def get_site_folders(project_root):
    
    """
    This function defines the four Oxford forest sites we are working with
    and builds the full folder path for each one. Each site folder is
    expected to contain a subfolder called individual_clouds with the raw
    .pcd scans inside it. Keeping this list in one place means that if a
    folder name ever changes, we only need to update it here.
    """
    
    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
    site_paths = {}
    for site_name in site_names:
        site_paths[site_name] = os.path.join(project_root, site_name)
    return site_paths





def compute_average_reduction_for_site(site_folder, voxel_sizes):
    
    """
    This function loads every single .pcd scan inside one site's
    individual_clouds folder, and for each voxel size we care about, works
    out what percentage of points were removed by voxelization for that
    specific scan. It keeps a running total of these percentages for each
    voxel size across all scans in the site, so that at the end we can
    divide by the number of scans and get the average reduction percentage
    for that site, at each voxel size. This tells us not just what happens
    to one or two example scans, but what we should generally expect across
    the whole site.
    """
    
    individual_clouds_path = os.path.join(site_folder, "individual_clouds")

    if not os.path.isdir(individual_clouds_path):
        return None

    all_files = [f for f in os.listdir(individual_clouds_path) if f.endswith(".pcd")]
    total_files = len(all_files)

    if total_files == 0:
        return None

    reduction_totals = {voxel_size: 0.0 for voxel_size in voxel_sizes}

    for file_name in all_files:
        file_path = os.path.join(individual_clouds_path, file_name)
        original_cloud = o3d.io.read_point_cloud(file_path)
        original_count = len(original_cloud.points)

        if original_count == 0:
            continue

        for voxel_size in voxel_sizes:
            downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)
            downsampled_count = len(downsampled_cloud.points)
            reduction_percentage = (1 - (downsampled_count / original_count)) * 100
            reduction_totals[voxel_size] += reduction_percentage

    average_reductions = {
        voxel_size: (reduction_totals[voxel_size] / total_files)
        for voxel_size in voxel_sizes
    }

    return average_reductions, total_files








def print_summary_table(all_site_results, voxel_sizes):
    """
    This function takes the average reduction results we calculated for
    every site and prints them out as a simple, readable table, with one
    row per site and one column per voxel size. Having everything laid out
    together like this makes it much easier to compare sites and voxel
    sizes side by side, and to spot whether one voxel size is consistently
    the best balance across all four forests, or whether the right choice
    actually depends on the specific site.
    """
    print("\nAverage Point Reduction Summary is as follows:")
    print("\n")

    header = f"{'Site':<22}" + "".join(f"{f'{v}m':>12}" for v in voxel_sizes)
    print(header)
    print("\n")

    for site_name, (average_reductions, total_files) in all_site_results.items():
        row = f"{site_name:<22}"
        for voxel_size in voxel_sizes:
            row += f"{average_reductions[voxel_size]:>11.1f}%"
        print(row)
        print(f"  (based on {total_files} scans)")

    print("\n \n \n")





if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_sizes_to_test = [0.05, 0.10, 0.20]

    site_folders = get_site_folders(project_root)
    all_site_results = {}

    for site_name, site_folder in site_folders.items():
        print(f"\nProcessing site: {site_name}")
        result = compute_average_reduction_for_site(site_folder, voxel_sizes_to_test)

        if result is None:
            print(f"Skipping {site_name}: no individual_clouds folder or no .pcd files found")
            continue

        average_reductions, total_files = result
        all_site_results[site_name] = (average_reductions, total_files)

        for voxel_size in voxel_sizes_to_test:
            print(f"  Voxel size {voxel_size}m: {average_reductions[voxel_size]:.1f}% average reduction")

    if all_site_results:
        print_summary_table(all_site_results, voxel_sizes_to_test)
    else:
        print("\n No sites were processed. Check that individual_clouds folders exist and contain .pcd files.")
        
        
        
        
        
        
        
        
# Processing site: wytham_data
#   Voxel size 0.05m: 41.7% average reduction
#   Voxel size 0.1m: 49.7% average reduction
#   Voxel size 0.2m: 62.1% average reduction

# Processing site: stein-am-rhein_data
#   Voxel size 0.05m: 48.2% average reduction
#   Voxel size 0.1m: 59.0% average reduction
#   Voxel size 0.2m: 71.8% average reduction

# Processing site: forest-of-dean_data
#   Voxel size 0.05m: 47.9% average reduction
#   Voxel size 0.1m: 56.8% average reduction
#   Voxel size 0.2m: 69.1% average reduction

# Processing site: evo_data
#   Voxel size 0.05m: 46.1% average reduction
#   Voxel size 0.1m: 54.8% average reduction
#   Voxel size 0.2m: 66.5% average reduction



# Average Point Reduction Summary

# Site                         0.05m        0.1m        0.2m


# wytham_data                  41.7%       49.7%       62.1%
#   (based on 625 scans)
# stein-am-rhein_data          48.2%       59.0%       71.8%
#   (based on 463 scans)
# forest-of-dean_data          47.9%       56.8%       69.1%
#   (based on 625 scans)
# evo_data                     46.1%       54.8%       66.5%
#   (based on 1002 scans)