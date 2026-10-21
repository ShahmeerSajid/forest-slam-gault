"""
LoGG3D-Net Training Pickle Builder
Forest SLAM Project - Oxford Dataset

LoGG3D-Net does not read our loop_closure_pairs.csv files when it trains.
It reads a "training pickle", which is a Python file holding one record for
every scan, and each record lists which other scans are the same place as
this one (its positives) and which other scans are close enough that they
must never be used as different places (its non-negatives). Every scan not
in the non-negatives list is treated as a different place automatically.
This script builds that pickle from the slam_poses.csv file and the
voxelized scans of the Oxford sites we choose to train on.

To keep the test fair, the site we want to test on is held out and never
goes into the pickle. For example, holding out wytham_data builds a pickle
from Stein-am-Rhein, Forest of Dean and Evo only, so a model fine-tuned on
it can be tested on Wytham, which it has never seen. All the training
sites go into one single pickle, because LoGG3D-Net's loader takes one
pickle per dataset name and we use the name "bushwalk" for it.

The positive distance and the non-negative distance follow the authors'
own KITTI settings, 3 m and 20 m, which are looser than the 1.5 m we use to
label our test pairs. Training wants plenty of easy and medium examples, and
the test labels stay strict. A positive here may be a neighbouring scan in
time, which is the authors' convention too. The script prints how many scans
also have a real revisit partner, which is the useful kind of positive.

How to run it:
    python3 make_logg3d_training_pickle.py --hold-out wytham_data
    python3 make_logg3d_training_pickle.py --hold-out none   (train on all four sites)

The scans themselves are not copied. The pickle only stores their relative
paths, so the training machine needs the four site folders, and the
LoGG3D-Net training setting bushwalk_dir must point at the folder that
contains them.
"""

import argparse
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd


TRAINING_TUPLE_MODULE = "utils.data_loaders.general.general_dataset"
ALL_SITES = ["wytham_data", "stein-am-rhein_data", "forest-of-dean_data", "evo_data"]


def build_training_tuple_class():
    """
    This function creates the record class that LoGG3D-Net expects to find
    inside a training pickle and registers it under the same module name the
    authors use. A pickle remembers the module path of each class it stores,
    and the training code will look for the class at exactly that path when
    it loads the file. Defining the class here under that path lets this
    script write a correct pickle on any machine, without needing the
    LoGG3D-Net repository, a GPU, or its heavy libraries to be installed.
    """
    class TrainingTuple:
        def __init__(self, id, timestamp, rel_scan_filepath, positives,
                     non_negatives, pose, positives_poses=None):
            self.id = id
            self.timestamp = timestamp
            self.rel_scan_filepath = rel_scan_filepath
            self.positives = positives
            self.non_negatives = non_negatives
            self.pose = pose
            self.positives_poses = positives_poses

    TrainingTuple.__module__ = TRAINING_TUPLE_MODULE
    TrainingTuple.__qualname__ = "TrainingTuple"

    name_parts = TRAINING_TUPLE_MODULE.split(".")
    for depth in range(1, len(name_parts) + 1):
        partial_name = ".".join(name_parts[:depth])
        if partial_name not in sys.modules:
            sys.modules[partial_name] = types.ModuleType(partial_name)
        if depth > 1:
            setattr(sys.modules[".".join(name_parts[:depth - 1])], name_parts[depth - 1],
                    sys.modules[partial_name])
    sys.modules[TRAINING_TUPLE_MODULE].TrainingTuple = TrainingTuple
    return TrainingTuple


def parse_command_line_arguments():
    """
    This function reads the settings given on the command line. The project
    root is the folder holding the four site folders, and it falls back to
    the parent of the script's folder when the sites are not next to the
    script, the same way our baseline scripts do. The hold-out setting names
    the site to leave out of training, or none to use all four. The voxel
    size picks which preprocessed scan folder is read, the two distances set
    how close scans must be to count as positives and as non-negatives, and
    the output setting is the folder the pickle is written to.
    """
    parser = argparse.ArgumentParser(
        description="Build a LoGG3D-Net training pickle from the Oxford sites."
    )
    parser.add_argument("--project-root", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Folder that contains the site folders.")
    parser.add_argument("--hold-out", default="wytham_data",
                        help="Site left out of training, or 'none' to use all four.")
    parser.add_argument("--voxel-size", type=float, default=0.10,
                        help="Which voxelized scan folder to read.")
    parser.add_argument("--positive-distance", type=float, default=3.0,
                        help="Scans closer than this (metres) are positives.")
    parser.add_argument("--non-negative-distance", type=float, default=20.0,
                        help="Scans closer than this (metres) are never used as negatives.")
    parser.add_argument("--revisit-gap", type=int, default=50,
                        help="Scan-number gap that makes a positive a real revisit (for reporting).")
    parser.add_argument("--output-folder", default=".", help="Where to write the pickle.")
    return parser.parse_args()


def find_data_root(project_root):
    """
    This function returns the folder that really contains the site folders.
    It uses the given project root when the first site folder is inside it,
    and otherwise tries the parent folder, so the script works both from the
    project root and from a subfolder such as Baselines.
    """
    if os.path.isdir(os.path.join(project_root, ALL_SITES[0])):
        return project_root
    parent_folder = os.path.dirname(project_root)
    if os.path.isdir(os.path.join(parent_folder, ALL_SITES[0])):
        return parent_folder
    return project_root


def read_site_poses(site_folder):
    """
    This function loads one site's slam_poses.csv and returns, in the
    order of the file, the scan names, the timestamps and a table of seven
    numbers per scan: x, y, z and the four quaternion values. LoGG3D-Net's
    general loader reads a pose as exactly these seven numbers in this order.
    If the quaternion columns are missing, it uses no rotation and prints a
    warning, because the loader then cannot line up neighbouring scans well.
    """
    poses_table = pd.read_csv(os.path.join(site_folder, "slam_poses.csv"))
    scan_names = poses_table["individual_cloud"].astype(str).values

    if "timestamp" in poses_table.columns:
        timestamps = poses_table["timestamp"].values
    else:
        timestamps = np.arange(len(poses_table))

    quaternion_columns = ["qx", "qy", "qz", "qw"]
    if all(column in poses_table.columns for column in quaternion_columns):
        quaternions = poses_table[quaternion_columns].values
    else:
        print(f"  Warning: no quaternion columns in {site_folder}; using no rotation.")
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(poses_table), 1))

    seven_numbers = np.hstack([poses_table[["x", "y", "z"]].values, quaternions]).astype(np.float64)
    return scan_names, timestamps, seven_numbers


def build_site_records(site_name, site_folder, voxel_size, id_offset, positive_distance,
                       non_negative_distance, revisit_gap, training_tuple_class):
    """
    This function builds the records for one site. For every scan it finds
    the other scans of the same site within the positive distance, which
    become its positives, and the ones within the non-negative distance,
    which become its non-negatives. If a scan has no other scan inside the
    positive distance, its single nearest scan is used instead, because the
    training loader needs at least one positive for every scan. Ids are
    shifted by id_offset so that the sites stack into one pickle with
    consecutive ids, and the positive and non-negative lists are kept as
    ids local to this pickle. The scan paths are relative to the folder
    that holds the sites. It returns the records, how many scans had to use
    the nearest-scan fallback, and how many scans have a real revisit
    partner.
    """
    
    
    scan_names, timestamps, poses = read_site_poses(site_folder)
    positions = poses[:, :3]
    voxel_folder_name = f"individual_clouds_voxel_{voxel_size:.2f}m"

    records = []
    fallback_count = 0
    revisit_count = 0

    for i in range(len(scan_names)):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        distances[i] = np.inf

        positive_ids = np.where(distances <= positive_distance)[0]
        if len(positive_ids) == 0:
            positive_ids = np.array([int(np.argmin(distances))])
            fallback_count += 1

        non_negative_ids = np.where(distances <= non_negative_distance)[0]
        non_negative_ids = np.union1d(non_negative_ids, positive_ids)

        if np.any(np.abs(positive_ids - i) >= revisit_gap):
            revisit_count += 1

        relative_path = os.path.join(site_name, voxel_folder_name, f"{scan_names[i]}.pcd")
        records.append(training_tuple_class(
            id=id_offset + i,
            timestamp=int(timestamps[i]),
            rel_scan_filepath=relative_path,
            positives=np.sort(positive_ids + id_offset),
            non_negatives=np.sort(non_negative_ids + id_offset),
            pose=poses[i],
        ))

    return records, fallback_count, revisit_count


def check_scan_files_exist(records, data_root, how_many=5):
    """
    This function checks that the scan files named in the records are really
    on disk by testing a few evenly spaced ones, and prints a warning if any
    is missing. This catches the most common mistake early, which is a
    wrong voxel size or a project root that is one folder off, instead of
    letting the training run fail hours later.
    """
    step = max(1, len(records) // how_many)
    for record in records[::step][:how_many]:
        full_path = os.path.join(data_root, record.rel_scan_filepath)
        if not os.path.isfile(full_path):
            print(f"  Warning: scan file not found: {full_path}")
            return False
    return True



if __name__ == "__main__":
    args = parse_command_line_arguments()
    training_tuple_class = build_training_tuple_class()
    data_root = find_data_root(os.path.abspath(args.project_root))

    held_out = None if args.hold_out.lower() == "none" else args.hold_out
    if held_out is not None and held_out not in ALL_SITES:
        raise SystemExit(f"--hold-out must be one of {ALL_SITES} or 'none'.")

    training_sites = [site for site in ALL_SITES if site != held_out]
    print(f"Data folder: {data_root}")
    print(f"Training sites: {training_sites}")
    print(f"Held out: {held_out}")

    all_records = []
    for site_name in training_sites:
        site_folder = os.path.join(data_root, site_name)
        if not os.path.isfile(os.path.join(site_folder, "slam_poses.csv")):
            raise SystemExit(f"No slam_poses.csv in {site_folder}")

        records, fallback_count, revisit_count = build_site_records(
            site_name, site_folder, args.voxel_size, len(all_records),
            args.positive_distance, args.non_negative_distance, args.revisit_gap,
            training_tuple_class,
        )
        files_ok = check_scan_files_exist(records, data_root)
        print(f"{site_name}: {len(records)} scans | with a real revisit partner: {revisit_count} | "
              f"nearest-scan fallback: {fallback_count} | sample files found: {files_ok}")
        all_records.extend(records)

    training_dictionary = {record.id: record for record in all_records}

    os.makedirs(args.output_folder, exist_ok=True)
    holdout_label = held_out if held_out else "none"
    output_path = os.path.join(args.output_folder, f"oxford_train_holdout_{holdout_label}.pickle")
    with open(output_path, "wb") as pickle_file:
        pickle.dump(training_dictionary, pickle_file)

    print(f"\nSaved {len(training_dictionary)} scan records to {output_path}")
