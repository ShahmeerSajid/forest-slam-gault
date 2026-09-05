
# Data Set + Data Loader:

# Pytorch needs the data in a consistent and organized way during training. Forexample if we have 
# 100,000 scan pairs, we could technically load everything into PyTorch, but that would be very inefficient and 
# difficult to manage. So Instead we are using DataSet, and DataLoader.

# DataSet -> tells Pytorch how to find and prepare one example.
# DataLoader -> tells PyTorch how to efficiently get many examples in batches.
# PyTorch/model -> trains on those batches.

# This code contains both the Dataset and DataLoader. The Dataset reads the CSV, 
# finds the two .pcd files, opens them, prepares each scan to the required number of 
# points, and returns (scan1, scan2, label). The DataLoader takes many of these examples 
# and puts them into batches. 


# The the training code takes each batch and gives it to the ML model. The model predicts whether the two scans are
# from the same place. PyTorch compares that prediction with the correct label, calculates 
# the error, and updates the model so it gets better. This repeats for many batches 
# until the model learns to recognize loop closures.


# So the pipeline is:
# PCD files + CSV → (Dataset → DataLoader) → PyTorch model → training

# The key idea is that PyTorch is not responsible for understanding the files. This DataSet/DataLoader code
# teaches PyTorch how to get the specific LiDAR data into a usable format. Then PyTorch handles the ML training.






import os
import numpy as np
import pandas as pd
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
 
 
 
 
class LoopClosureDataset(Dataset):
    

 
  
    def __init__(self, site_folder, voxel_size, max_points=19600):
        
        """
        This sets up the dataset for one specific site. It needs the path
        to that site's folder, the voxel size we want to use, since that
        determines which preprocessed folder of point clouds to read from,
        and a maximum number of points per scan. Point clouds do not all
        have exactly the same number of points even after voxelization, but
        PyTorch needs every example in a batch to have the same shape, so
        we fix a maximum point count here. Scans with more points than this
        will be randomly trimmed down, and scans with fewer points will be
        padded with extra copies of existing points, so that every scan we
        return always has exactly this many points.
        """
        
        self.site_folder = site_folder
        self.max_points = max_points
 
        voxel_folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"
        self.point_cloud_folder = os.path.join(site_folder, voxel_folder_name)
 
        labels_path = self._find_labels_file(site_folder)
        self.pairs_table = pd.read_csv(labels_path)
 
 
 
 
    def _find_labels_file(self, site_folder):
        """
        This helper function looks inside a site's folder for its
        loop_closure_pairs.csv file. Since earlier versions of our labeling
        script sometimes saved this file with a site-specific prefix and
        sometimes without one, this function checks for both possible names
        so the dataset can still find the right file either way, without
        needing the exact filename hardcoded.
        """
        site_name = os.path.basename(os.path.normpath(site_folder))
        possible_names = [
            f"{site_name}_loop_closure_pairs.csv",
            "loop_closure_pairs.csv",
        ]
 
        for name in possible_names:
            candidate_path = os.path.join(site_folder, name)
            if os.path.isfile(candidate_path):
                return candidate_path
 
        raise FileNotFoundError(
            f"No loop_closure_pairs.csv file found in {site_folder}"
        )
 
 
 
 
 
 
    def __len__(self):
        """
        This tells PyTorch how many examples are in this dataset, which is
        simply the number of rows in the loop_closure_pairs.csv file, since
        each row represents one scan pair example.
        """
        return len(self.pairs_table)
 
 
 
 
 
    def _load_and_prepare_scan(self, scan_name):
        """
        This function loads one single point cloud file from disk, given
        just its scan name, and converts it into a fixed-size NumPy array
        of x, y, z coordinates. If the scan has more points than our
        max_points limit, we randomly select a subset of that size. If it
        has fewer, we randomly duplicate some existing points until it
        reaches that size. This padding and trimming step is necessary
        because PyTorch requires every item in a batch to be the exact same
        shape, even though real point clouds naturally vary in size.
        """
        file_path = os.path.join(self.point_cloud_folder, f"{scan_name}.pcd")
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points, dtype=np.float32)
 
        num_points = points.shape[0]
 
        if num_points == 0:
            points = np.zeros((self.max_points, 3), dtype=np.float32)
            return points
 
        if num_points >= self.max_points:
            chosen_indices = np.random.choice(num_points, self.max_points, replace=False)
            points = points[chosen_indices]
        else:
            extra_needed = self.max_points - num_points
            extra_indices = np.random.choice(num_points, extra_needed, replace=True)
            extra_points = points[extra_indices]
            points = np.concatenate([points, extra_points], axis=0)
 
        return points
 
 
 
 
 
 
    def __getitem__(self, index):
        """
        This is the core function PyTorch calls whenever it wants a single
        training example. Given an index, it looks up that row in the
        loop_closure_pairs.csv table, finds the two scan names and the
        label for that row, loads and prepares both point clouds using the
        helper function above, and returns everything as PyTorch tensors,
        which is the format the rest of the training pipeline expects to
        work with.
        """
        row = self.pairs_table.iloc[index]
 
        scan_1_points = self._load_and_prepare_scan(row["scan_1"])
        scan_2_points = self._load_and_prepare_scan(row["scan_2"])
 
        scan_1_tensor = torch.from_numpy(scan_1_points)
        scan_2_tensor = torch.from_numpy(scan_2_points)
        label_tensor = torch.tensor(row["label"], dtype=torch.float32)
 
        return scan_1_tensor, scan_2_tensor, label_tensor
 
 
 
 
 
 
 
def create_dataloader_for_site(site_folder, voxel_size, batch_size=16, shuffle=True, max_points=19600):
    """
    This function is a convenient shortcut for creating a ready-to-use
    DataLoader for one site. It builds the LoopClosureDataset for that
    site, then wraps it in a PyTorch DataLoader, which takes care of
    grouping individual examples into batches of the given batch_size,
    optionally shuffling the order of examples each time through the data,
    which helps training generalize better rather than always seeing pairs
    in the same fixed order.
    """
    dataset = LoopClosureDataset(site_folder, voxel_size, max_points=max_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
 
 
 
 
 
 
 
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    voxel_size = 0.10
    max_points = 19500
 
    site_names = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]
 
    for site_name in site_names:
        site_folder = os.path.join(project_root, site_name)
 
        print(f"\nTesting site: {site_name}")
 
        try:
            dataloader = create_dataloader_for_site(
                site_folder, voxel_size, batch_size=4, max_points=max_points
            )
 
            print(f"  Dataset size: {len(dataloader.dataset)} pairs")
 
            scan_1_batch, scan_2_batch, label_batch = next(iter(dataloader))
 
            print(f"  Scan 1 batch shape: {scan_1_batch.shape}")
            print(f"  Scan 2 batch shape: {scan_2_batch.shape}")
            print(f"  Label batch: {label_batch}")
 
        except Exception as error:
            print(f"  Failed for {site_name}: {error}")
 
    print("\nAll sites tested.")