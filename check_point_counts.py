
"""

Before picking a fixed max_points value for the Dataset class (to feed into the PyTorch Model), 
we need to actually know how many points our voxelized scans typically have,
across all four sites. So this code checks the real point counts, so our
max_points decision is based on evidence rather than a guess.

"""






import os
import numpy as np
import pandas as pd
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader




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





def check_point_counts_for_site(site_folder, voxel_size):
    
    """
    This function looks inside a site's voxelized point cloud folder for a
    given voxel size, loads every scan, and records how many points each
    one has. It then reports the minimum, maximum, average, and median
    point count across all scans in that site. These numbers tell us the
    real range of scan sizes we need to design our fixed max_points value
    around, instead of guessing.
    """
    
    voxel_folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"
    voxel_folder_path = os.path.join(site_folder, voxel_folder_name)

    if not os.path.isdir(voxel_folder_path):
        return None

    all_files = [f for f in os.listdir(voxel_folder_path) if f.endswith(".pcd")]

    if len(all_files) == 0:
        return None

    point_counts = []
    for file_name in all_files:
        file_path = os.path.join(voxel_folder_path, file_name)
        point_cloud = o3d.io.read_point_cloud(file_path)
        point_counts.append(len(point_cloud.points))

    point_counts = np.array(point_counts)

    stats = {
        "min": int(np.min(point_counts)),
        "max": int(np.max(point_counts)),
        "mean": float(np.mean(point_counts)),
        "median": float(np.median(point_counts)),
        "total_scans": len(point_counts),
    }

    return stats



if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_size = 0.10

    site_folders = get_site_folders(project_root)
    all_stats = {}

    for site_name, site_folder in site_folders.items():
        stats = check_point_counts_for_site(site_folder, voxel_size)

        if stats is None:
            print(f"Skipping {site_name}: no voxelized folder found for {voxel_size}m")
            continue

        all_stats[site_name] = stats

        print(f"\nSite: {site_name} ({stats['total_scans']} scans)")
        print(f"  Min points: {stats['min']}")
        print(f"  Max points: {stats['max']}")
        print(f"  Mean points: {stats['mean']:.0f}")
        print(f"  Median points: {stats['median']:.0f}")




    if all_stats:
        overall_min = min(s["min"] for s in all_stats.values())
        overall_max = max(s["max"] for s in all_stats.values())
        overall_mean = np.mean([s["mean"] for s in all_stats.values()])

        print("\n")
        print("Overall across all sites:")
        print(f"  Smallest scan anywhere: {overall_min} points")
        print(f"  Largest scan anywhere: {overall_max} points")
        print(f"  Average of site means: {overall_mean:.0f} points")
     
      
      
      
      
      
      
      
      
    
    
    
# Site: wytham_data (625 scans)
#   Min points: 25133
#   Max points: 37103
#   Mean points: 31650
#   Median points: 31566

# Site: stein-am-rhein_data (463 scans)
#   Min points: 19969
#   Max points: 58115
#   Mean points: 49712
#   Median points: 51190

# Site: forest-of-dean_data (625 scans)
#   Min points: 19622
#   Max points: 28195
#   Mean points: 23000
#   Median points: 22748

# Site: evo_data (1002 scans)
#   Min points: 37971
#   Max points: 59817
#   Mean points: 51884
#   Median points: 51705


# Overall across all sites:
#   Smallest scan anywhere: 19622 points
#   Largest scan anywhere: 59817 points
#   Average of site means: 39061 points



# CONCLUSION:


# From the code I saw that the LiDAR scans have different numbers of points:
# Wytham: around 25,000+
# Forest of Dean: around 20,000–28,000
# Stein am Rhein: around 50,000+
# Evo: around 50,000+

# Since PyTorch requires every scan to have the exact same number of points to work with, 
# we needed to pick one fixed number that works across all four sites. We chose around 
# 19,600 because it is close to the smallest scan found anywhere in the 
# dataset (Forest of Dean's minimum), which means every single scan, even the smallest one, 
# already has enough real points to randomly trim down to that size, without ever 
# needing to add fake duplicate points to make up the difference. This keeps our 
# data honest and consistent across all sites, which matters since we are directly 
# comparing performance across different forests, and avoids introducing any artificial 
# padding that could subtly distort the model's understanding of sparser scans.

# Because we train one model using all four forests together, we want the model to 
# learn what a place looks like in 3D, not differences in how many points each forest 
# happens to have. If Evo always has 40,000 points forexample while another forest has 20,000, 
# the model might accidentally use that difference as a clue about which forest the 
# scan came from. 
# Gault uses a different LiDAR sensor, so its scans may have a different number of points 
# than the four Oxford forests. If the model accidentally learns that 
# “a certain number of points usually means a certain type of place,” 
# it might rely on that instead of actually learning the shape and arrangement 
# of the trees. When we give it Gault, that point density will be different from 
# what it saw during training, so that could hurt 
# its loop-closure performance. That's why we want the model to focus on the actual 
# 3D structure of the forest, not the number of points in the scan.



# IMPORTANT to note that we still needed voxelization. 
# Voxelization is a smarter first step than randomly cutting the raw scans. The original 
# LiDAR scans had around 60,000–70,000+ points, so if we directly reduced them to around 
# 20,000 points, we would have to remove a very large percentage of the data randomly. 
# That could accidentally remove important points, such as parts of trees, branches, or 
# other structures, and create uneven gaps. Voxelization first organizes the point cloud
# spatially by dividing the space into small 3D boxes (voxels) and keeping a representative 
# point from each area. This reduces the number of points while maintaining the overall 
# shape and structure of the forest. Voxelization intelligently reduced the raw scans first,
# merging nearby points based on spatial density so the OVERALL SHAPE stayed evenly and
# faithfully represented.

# Then, reducing the voxelized scan further to 19,600 points
# is a much smaller and safer reduction because the scan has already been 
# intelligently simplified.