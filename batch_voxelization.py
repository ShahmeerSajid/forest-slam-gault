"""
Batch Voxelization Script (Multi-Site Version)
Forest SLAM Project - Oxford Dataset Preprocessing
 
This script processes every .pcd scan across all four Oxford forest sites
(Wytham, Stein am Rhein, Forest of Dean, Evo) and saves voxelized versions
of each scan into folders inside that same site's own data folder. Each
site is handled completely separately, since scans from different forests
should never be mixed together when we later build loop-closure pairs.
"""
 
 
 
import open3d as o3d
import os
import numpy as np
import pandas as pd
 
 
 
 
def get_site_folders(project_root):
    """
    This function defines the four Oxford forest sites we are working with
    and builds the full folder path for each one, based on where the
    project lives on disk. Each site folder is expected to contain a
    subfolder called individual_clouds with the raw .pcd scans inside it,
    along with a slam_poses.csv file. Keeping this list in one place means
    that if a folder name ever changes, we only need to update it here
    rather than throughout the rest of the script.
    """
    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
    site_paths = {}
    for site_name in site_names:
        site_paths[site_name] = os.path.join(project_root, site_name)
    return site_paths
 
 
 
 
 
def create_output_folders(site_folder, voxel_sizes):
    """
    This function creates one output folder for each voxel size we want to
    generate, placed inside the given site's own data folder, right next to
    its individual_clouds folder. This keeps every version of a site's data,
    raw and voxelized, together in one place, rather than mixed in with
    other sites.
    """
    output_folders = {}
    for voxel_size in voxel_sizes:
        folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"
        folder_path = os.path.join(site_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        output_folders[voxel_size] = folder_path
    return output_folders
 
 
 
 
 
def voxelize_and_save_all_scans(source_folder, output_folders, voxel_sizes, site_label):
    """
    This function goes through every .pcd file inside one site's
    individual_clouds folder, loads it, and for each voxel size we care
    about, applies voxel downsampling and saves the result into the
    matching output folder using the same file name as the original. The
    site_label is only used for printing progress messages, so it is clear
    in the terminal which forest is currently being processed, since this
    will run for all four sites one after another.
    """
    if not os.path.isdir(source_folder):
        print(f"Skipping {site_label}: no individual_clouds folder found at {source_folder}")
        return
 
    all_files = [f for f in os.listdir(source_folder) if f.endswith(".pcd")]
    total_files = len(all_files)
 
    if total_files == 0:
        print(f"Skipping {site_label}: no .pcd files found in {source_folder}")
        return
 
    print(f"\nProcessing site: {site_label} ({total_files} scans)")
 
    for index, file_name in enumerate(all_files, start=1):
        source_path = os.path.join(source_folder, file_name)
        original_cloud = o3d.io.read_point_cloud(source_path)
 
        for voxel_size in voxel_sizes:
            downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)
            output_path = os.path.join(output_folders[voxel_size], file_name)
            o3d.io.write_point_cloud(output_path, downsampled_cloud)
 
        if index % 25 == 0 or index == total_files:
            print(f"  {site_label}: processed {index}/{total_files}")
 
    print(f"Finished site: {site_label}")
 
 
 
 
if __name__ == "__main__":
    project_root = "/Users/shahmeer/forest-slam-project"
    voxel_sizes_to_generate = [0.05, 0.10, 0.20]
 
    site_folders = get_site_folders(project_root)
 
    for site_label, site_folder in site_folders.items():
        individual_clouds_path = os.path.join(site_folder, "individual_clouds")
        output_folders = create_output_folders(site_folder, voxel_sizes_to_generate)
        voxelize_and_save_all_scans(
            individual_clouds_path, output_folders, voxel_sizes_to_generate, site_label
        )
 
    print("\nAll sites complete.")
    
    
    
    
    
# Note:   
# slam_poses.csv files don't need to be touched or duplicated because they store each scan's 
# position/orientation, not point data, so voxelization doesn't affect it at all. 

# The original slam_poses.csv files apply to the original scans and all three voxelized
# versions equally, since voxelizing a scan doesn't move where that scan was taken from.
# So: one slam_poses.csv, shared across all folders, no changes needed there.

# And we have create 3 more folders now inside the project repo FOR EACH OXFORD FOREST - each folder corresponds
# to a batch of (0.05m , 0.1m and 0.2m) voxelizations.