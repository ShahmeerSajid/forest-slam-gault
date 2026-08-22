"""
Voxelization Comparison Script (Multi-Site Version)
Forest SLAM Project - Oxford Dataset Preprocessing Exploration
 
This script lets us test and visually compare voxel downsampling on a few
sample scans from any one of the four Oxford forest sites, by simply
changing which site we want to look at.
"""
 
 
 
import open3d as o3d
import numpy as np
import pandas as pd
import os
 
 
 
 
 
def load_point_cloud(file_path):
    """
    This function takes the file path of a single .pcd scan and loads it
    into memory using Open3D. We wrap this loading step in its own function
    so we can reuse it every time we want to bring in a new scan.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud
 
 
 
 
def count_points(point_cloud):
    """
    This function counts how many individual points exist inside a given
    point cloud, which we use before and after voxelization to see exactly
    how much the point cloud has shrunk.
    """
    points_array = np.asarray(point_cloud.points)
    return len(points_array)
 
 
 
 
def apply_voxel_downsampling(point_cloud, voxel_size):
    """
    This function divides 3D space into a grid of small cubes, called
    voxels, where each cube has a side length equal to voxel_size in
    meters. Every point that falls inside the same cube gets replaced by a
    single representative point. A larger voxel_size means bigger cubes,
    which means more points get merged together, giving a smaller but less
    detailed point cloud. A smaller voxel_size keeps more detail but
    reduces the point count less.
    """
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_cloud
 
 
 
 
 
 
def compare_voxel_sizes(file_path, voxel_sizes):
    """
    This function loads one scan, counts its original points, then applies
    voxel downsampling at each voxel size we want to test, printing a
    before-and-after comparison so we can see how much each voxel size
    reduces the scan.
    """
    original_cloud = load_point_cloud(file_path)
    original_count = count_points(original_cloud)
 
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Original point count: {original_count}")
 
    results = {}
    
    for voxel_size in voxel_sizes:
        downsampled_cloud = apply_voxel_downsampling(original_cloud, voxel_size)
        downsampled_count = count_points(downsampled_cloud)
        reduction_percentage = (1 - (downsampled_count / original_count)) * 100
 
        print(
            f"Voxel size {voxel_size}m -> "
            f"{downsampled_count} points "
            f"({reduction_percentage:.1f}% reduction)"
        )
 
        results[voxel_size] = downsampled_cloud
 
    return original_cloud, results
 
 
 
 
 
 
def visualize_comparison(original_cloud, downsampled_cloud, voxel_size):
    """
    This function opens two separate visual windows at the same time, one
    showing the original scan and one showing the voxelized version, with
    full zoom, rotate, and pan working normally in both. Since macOS only
    allows visual windows to run on a program's main thread, we save each
    point cloud to a temporary file and launch two separate Python
    processes, each controlling its own independent window, so both work
    fully and at the same time.
    """
    import subprocess
    import tempfile
 
    temp_dir = tempfile.mkdtemp()
    original_path = os.path.join(temp_dir, "original.pcd")
    downsampled_path = os.path.join(temp_dir, "downsampled.pcd")
 
    o3d.io.write_point_cloud(original_path, original_cloud)
    o3d.io.write_point_cloud(downsampled_path, downsampled_cloud)
 
    viewer_script = os.path.join(temp_dir, "viewer.py")
    with open(viewer_script, "w") as f:
        f.write(
            "import open3d as o3d\n"
            "import sys\n"
            "cloud = o3d.io.read_point_cloud(sys.argv[1])\n"
            "o3d.visualization.draw_geometries([cloud], window_name=sys.argv[2])\n"
        )
 
    print(
        f"\nOpening two independent windows: original and "
        f"voxelized at {voxel_size}m. Close both windows when done..."
    )
 
    process_original = subprocess.Popen(
        ["python3", viewer_script, original_path, "Original Scan"]
    )
    process_voxelized = subprocess.Popen(
        ["python3", viewer_script, downsampled_path, f"Voxelized at {voxel_size}m"]
    )
 
    process_original.wait()
    process_voxelized.wait()
 
 
 
 
 
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
 
    site_to_test = "stein-am-rhein_data" # Important ---> WE HAVE 4 different forest to test from!
 
    dataset_folder = os.path.join(project_root, site_to_test, "individual_clouds")
 
    all_files = [f for f in os.listdir(dataset_folder) if f.endswith(".pcd")]
    test_files = all_files[:5]  # takes first 5 from each forest pcd for comparison ...
 
    voxel_sizes_to_test = [0.05, 0.10, 0.20]
 
    print(f"\n \nForest site: {site_to_test}")
    
    for file_name in test_files:
        file_path = os.path.join(dataset_folder, file_name)
        original_cloud, downsampled_results = compare_voxel_sizes(
            file_path, voxel_sizes_to_test
        )
 
    print("\nComparison complete for all test files.")
    
    print(
        "To visually inspect a specific scan and voxel size, call "
        "visualize_comparison() directly with the returned clouds."
    )
    
    size = 0.2
    visualize_comparison(original_cloud, downsampled_results[size], size)
    