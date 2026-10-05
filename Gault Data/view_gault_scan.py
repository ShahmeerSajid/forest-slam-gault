"""
Gault Scan Viewer Script
Forest SLAM Project - Gault Dataset
 
After extracting Gault's LiDAR scans out of the ROS2 bag, we need a quick
way to actually look at one and sanity check that it contains a real,
recognizable chunk of forest, rather than trusting the extraction blindly.
This script opens a single .pcd file from the extracted Gault scans and
displays it in a 3D viewer window, the same way we previously checked
scans from the Oxford sites.
"""


 
import os
import open3d as o3d
 
 
 
 
 
def list_available_scans(individual_clouds_folder):
    """
    This function looks inside the extracted Gault scans folder and
    returns a sorted list of all the .pcd file names found there. We use
    this so the script can show a few available file names on screen,
    which makes it easy to pick one to view without having to leave the
    script and dig through the folder manually.
    """
    all_files = [f for f in os.listdir(individual_clouds_folder) if f.endswith(".pcd")]
    all_files.sort()
    return all_files
 
 
 
 
 
 
 
def load_and_describe_scan(file_path):
    """
    This function loads a single .pcd file from disk using Open3D and
    prints out some basic information about it, such as how many points
    it contains. Printing the point count before opening the viewer
    window gives us an immediate, numeric sanity check, since an empty or
    broken scan would show zero or a suspiciously tiny point count even
    before we look at it visually.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"Loaded: {file_path}")
    print(f"Number of points: {len(point_cloud.points)}")
    return point_cloud
 
 
 
 
 
 
 
def show_scan_in_window(point_cloud, window_title):
    """
    This function opens an interactive 3D viewer window showing the scan,
    using Open3D's built-in visualizer. Inside this window, the scan can
    be rotated, zoomed, and panned with the mouse, which lets us visually
    confirm that the points form a recognizable shape, such as trees,
    ground, and scattered forest clutter, rather than random noise or an
    empty cloud.
    """
    
    o3d.visualization.draw_geometries([point_cloud], window_name=window_title)
 
 
 
 
if __name__ == "__main__":
    individual_clouds_folder = "individual_clouds"
 
    available_scans = list_available_scans(individual_clouds_folder)
 
    if len(available_scans) == 0:
        print(f"No .pcd files found in {individual_clouds_folder}")
    else:
        print(f"Found {len(available_scans)} scans. Showing the first 5 file names:")
        for file_name in available_scans[:5]:
            print(f"  {file_name}")
 
        scan_to_view = available_scans[10]
        file_path = os.path.join(individual_clouds_folder, scan_to_view)
 
        point_cloud = load_and_describe_scan(file_path)
        show_scan_in_window(point_cloud, window_title=scan_to_view)