"""
Voxelization Comparison Script
Forest SLAM Project - Oxford Dataset Preprocessing Exploration

This script helps us understand what voxelization actually does to our LiDAR
scans before we decide whether to use it as part of our preprocessing pipeline.
"""

import open3d as o3d
import numpy as np
import pandas as pd
import os


def load_point_cloud(file_path):
    """
    This function takes the file path of a single .pcd scan and loads it into
    memory using Open3D, which is the library we have been using to view our
    point clouds. A point cloud, once loaded, is essentially a big list of
    individual 3D points, where each point has an x, y, and z coordinate
    describing where it sits in space relative to the sensor. We wrap this
    loading step in its own function so that we can reuse it every time we
    want to bring in a new scan, instead of repeating the same lines of code
    throughout the script.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud


def count_points(point_cloud):
    """
    This function simply counts how many individual points exist inside a
    given point cloud. We will use this before and after voxelization to see
    exactly how much the point cloud has shrunk. Open3D stores the points as
    a NumPy array, so we can just check the length of that array to get the
    total count.
    """
    points_array = np.asarray(point_cloud.points)
    return len(points_array)


def apply_voxel_downsampling(point_cloud, voxel_size):
    """
    This is the core function of the whole script. Voxelization works by
    dividing the entire 3D space that the point cloud occupies into a grid
    of small cubes, where each cube is called a voxel. The size of each cube
    is controlled by the voxel_size parameter, measured in meters. So a
    voxel_size of 0.1 means each cube is 10 centimeters wide, tall, and deep.
    Once the grid is created, every point that falls inside the same cube
    gets replaced by a single point that represents the average position of
    all the original points in that cube. The practical effect is that dense
    clusters of points get thinned out into a single representative point,
    while empty areas of the scan stay empty. A larger voxel_size means
    bigger cubes, which means more points get merged together, which means
    a smaller but less detailed point cloud. A smaller voxel_size keeps more
    detail but does not reduce the point count as much.
    """
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_cloud





# Open the scans (original, modified) side by side and comapre using zoom/scan/slide.

def compare_voxel_sizes(file_path, voxel_sizes):
    """
    This function ties everything together for a single scan. It loads the
    original point cloud, counts its points, then applies voxel downsampling
    at each voxel size we want to test, counting the points after each one.
    The goal is to print out a clear before-and-after comparison so we can
    see, in plain numbers, how much each voxel size reduces the scan and
    start building an intuition for which voxel size might be a reasonable
    balance between reducing data size and keeping enough detail for later
    steps like ground-truth labeling and place recognition.
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






# def visualize_comparison(original_cloud, downsampled_cloud, voxel_size):
#     """
#     This function opens two separate visual windows at the same time, one
#     showing the original scan and one showing the voxelized version, so both
#     are visible on screen simultaneously instead of one appearing only after
#     the other is closed. Open3D's simple draw_geometries function normally
#     blocks the program and only shows one window at a time, so here we
#     manually create two independent Visualizer windows, position them next
#     to each other on screen, and then update both of them together in a
#     loop until you close them. This gives you two fully separate, movable
#     windows that are both open and interactive at once, which makes side by
#     side comparison much easier.
#     """
#     window_original = o3d.visualization.Visualizer()
#     window_original.create_window(
#         window_name="Original Scan", width=800, height=600, left=0, top=50
#     )
#     window_original.add_geometry(original_cloud)
 
#     window_voxelized = o3d.visualization.Visualizer()
#     window_voxelized.create_window(
#         window_name=f"Voxelized at {voxel_size}m",
#         width=800,
#         height=600,
#         left=820,
#         top=50,
#     )
#     window_voxelized.add_geometry(downsampled_cloud)
 
#     print(
#         f"\nTwo windows should now be open side by side: original and "
#         f"voxelized at {voxel_size}m. Close both windows to continue..."
#     )
 
#     keep_running = True
#     while keep_running:
#         keep_running_original = window_original.poll_events()
#         window_original.update_renderer()
 
#         keep_running_voxelized = window_voxelized.poll_events()
#         window_voxelized.update_renderer()
 
#         keep_running = keep_running_original and keep_running_voxelized
 
#     window_original.destroy_window()
#     window_voxelized.destroy_window()
    
    
    
    
def visualize_comparison(original_cloud, downsampled_cloud, voxel_size):
    """
    This function opens two separate visual windows at the same time, one
    showing the original scan and one showing the voxelized version, with
    full zoom, rotate, and pan working normally in both. On macOS, a visual
    window can only run on the main thread of a program, which is why
    running two windows in the same program at once either crashes or
    breaks mouse controls. To get around this, we save each point cloud to
    a temporary file on disk, and then launch two completely separate
    Python processes, each one opening and controlling its own single
    window. Since each process has its own main thread, both windows work
    fully and independently, at the same time, exactly as if you had opened
    each one on its own.
    """
    import subprocess
    import tempfile
    import os as os_module
 
    temp_dir = tempfile.mkdtemp()
    original_path = os_module.path.join(temp_dir, "original.pcd")
    downsampled_path = os_module.path.join(temp_dir, "downsampled.pcd")
 
    o3d.io.write_point_cloud(original_path, original_cloud)
    o3d.io.write_point_cloud(downsampled_path, downsampled_cloud)
 
    viewer_script = os_module.path.join(temp_dir, "viewer.py")
    with open(viewer_script, "w") as f:
        f.write(
            "import open3d as o3d\n"
            "import sys\n"
            "cloud = o3d.io.read_point_cloud(sys.argv[1])\n"
            "o3d.visualization.draw_geometries([cloud], window_name=sys.argv[2])\n"
        )
 
    print(
        f"\nOpening two independent windows: original and "
        f"voxelized at {voxel_size}m. Both support full zoom, rotate, and "
        f"pan. Close both windows when you are done comparing..."
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
    dataset_folder = "/Users/shahmeer/forest-slam-project/individual_clouds"

    all_files = [f for f in os.listdir(dataset_folder) if f.endswith(".pcd")]
    test_files = all_files[:3]

    voxel_sizes_to_test = [0.05, 0.10, 0.20]

    for file_name in test_files:
        file_path = os.path.join(dataset_folder, file_name)
        original_cloud, downsampled_results = compare_voxel_sizes(
            file_path, voxel_sizes_to_test
        )
    
    visualize_comparison(original_cloud, downsampled_results[0.1], 0.1)

    print("\nComparison complete for all test files.")
    print(
        "If you would like to visually inspect a specific scan and voxel size, "
        "call visualize_comparison() directly with the returned clouds."
    )
    
    
    
    
