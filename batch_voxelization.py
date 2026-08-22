"""
Batch Voxelization Code
Forest SLAM Project - Oxford Dataset Preprocessing

This script processes every .pcd scan in the Oxford dataset and saves a
voxelized version of each one into three separate folders, one for each
voxel size we have been testing (0.05m, 0.10m abd 0.20m). 

The original raw scans are never modified
or deleted, so we always keep the source data intact.

"""

import open3d as o3d
import pandas as pd
import numpy as np

import os


def create_output_folders(base_folder, voxel_sizes):
    """
    This function creates one output folder for each voxel size we want to
    generate, named clearly so it is obvious what is inside each one. For
    example, a voxel size of 0.1 will get a folder called
    individual_clouds_voxel_0.10m. If a folder already exists, we simply
    reuse it rather than raising an error, since we may want to rerun this
    script more than once.
    """
    output_folders = {}
    for voxel_size in voxel_sizes:
        folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"
        folder_path = os.path.join(os.path.dirname(base_folder), folder_name)
        os.makedirs(folder_path, exist_ok=True)
        output_folders[voxel_size] = folder_path
    return output_folders




def voxelize_and_save_all_scans(source_folder, output_folders, voxel_sizes):
    """
    This function goes through every single .pcd file in the source folder,
    loads it, and for each of the voxel sizes we care about, applies voxel
    downsampling and saves the result into the matching output folder, using
    the exact same file name as the original. This means that after running
    this function, each output folder will contain a full voxelized copy of
    the entire dataset at one specific voxel size, and the file names will
    line up across folders, making it easy to compare the same scan at
    different voxel sizes later, or to load whichever version we decide to
    use for training.
    """
    all_files = [f for f in os.listdir(source_folder) if f.endswith(".pcd")]
    total_files = len(all_files)

    for index, file_name in enumerate(all_files, start=1):
        source_path = os.path.join(source_folder, file_name)
        original_cloud = o3d.io.read_point_cloud(source_path)

        for voxel_size in voxel_sizes:
            downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)
            output_path = os.path.join(output_folders[voxel_size], file_name)
            o3d.io.write_point_cloud(output_path, downsampled_cloud)

        print(f"Processed {index}/{total_files}: {file_name}")

    print("\nAll scans have been voxelized and saved.")
    for voxel_size, folder_path in output_folders.items():
        print(f"Voxel size {voxel_size}m saved to: {folder_path}")




if __name__ == "__main__":
    source_folder = "/Users/shahmeer/forest-slam-project/individual_clouds"
    voxel_sizes_to_generate = [0.05, 0.10, 0.20]

    output_folders = create_output_folders(source_folder, voxel_sizes_to_generate)
    voxelize_and_save_all_scans(source_folder, output_folders, voxel_sizes_to_generate)
    
    
    
    
    
# Note:   
# slam_poses.csv doesn't need to be touched or duplicated because it stores each scan's 
# position/orientation, not point data, so voxelization doesn't affect it at all. 

# The same slam_poses.csv applies to the original scans and all three voxelized
# versions equally, since voxelizing a scan doesn't move where that scan was taken from.
# So: one slam_poses.csv, shared across all folders, no changes needed there.

# And we have create 3 more folders now inside the project repo - each folder corresponds
# to a batch of (0.05m , 0.1m and 0.2m) voxelizations.