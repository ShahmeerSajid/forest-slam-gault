"""

GNSS to KISS-ICP Alignment Script
How close are the poses collected using KISS-ICP to the poses collected using Satellite Antenna (.gnss) to each other. 
Forest SLAM Project - Gault Dataset

For one Gault drive for the rosbag: rosbag2_2026_10_06-20_10_27
we have two position tracks, the KISS-ICP poses
calculated from the LiDAR and the GNSS poses from the satellite receiver.

When the two are drawn on top of each other they do not match, because the
GNSS file is a mirror image of the KISS-ICP file, as if one of its axes
had its sign changed. A rotation and a shift can never turn a mirror image
into the original, so this script first flips the sign of one chosen axis
of the GNSS track, then finds the single rotation and shift that places it
best on top of the KISS-ICP track, prints how far apart the two tracks are
after that, and saves a picture with the tracks before and after.


Flipping x and flipping y give the same fit, only turned half a circle
from each other, so the position data alone cannot tell which of the two
the GNSS file really used. Both are fine for placing the drives in one
frame. Only the x and y positions are used, since the heights of the two
tracks disagree a lot and GNSS height is usually the weakest part.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_command_line_arguments():
    """
    This function reads the settings given on the command line. The kiss and
    gnss settings are the two pose files of one drive, which must have the
    same columns as our other pose files, with a timestamp column and x and
    y positions. The flip axis says which axis of the GNSS track gets its
    sign changed before the fit, and it is x unless told otherwise. The
    output setting is the name of the picture to save.
    """
    parser = argparse.ArgumentParser(
        description="Place a mirrored GNSS track on top of the KISS-ICP track of the same drive."
    )
    parser.add_argument("--kiss", required=True, help="KISS-ICP pose CSV of the drive.")
    parser.add_argument("--gnss", required=True, help="GNSS pose CSV of the same drive.")
    parser.add_argument("--flip-axis", choices=["x", "y"], default="x",
                        help="Which GNSS axis gets its sign changed.")
    parser.add_argument("--output", default="gnss_aligned_to_kiss.png",
                        help="Name of the picture to save.")
    return parser.parse_args()


def load_matching_tracks(kiss_path, gnss_path):
    """
    This function loads the two pose files and keeps only the poses whose
    timestamps appear in both files, in time order, so that row i of one
    track and row i of the other describe the same moment. It returns the x
    and y positions of each track as two arrays of the same length. It
    stops with a clear message if the two files share no timestamps,
    because then they cannot be from the same drive.
    """
    kiss_table = pd.read_csv(kiss_path)
    gnss_table = pd.read_csv(gnss_path)

    shared_timestamps = np.intersect1d(kiss_table.timestamp.values, gnss_table.timestamp.values)
    if len(shared_timestamps) == 0:
        raise ValueError("The two files share no timestamps, so they are not from the same drive.")

    kiss_xy = kiss_table.set_index("timestamp").loc[shared_timestamps][["x", "y"]].values
    gnss_xy = gnss_table.set_index("timestamp").loc[shared_timestamps][["x", "y"]].values
    return kiss_xy, gnss_xy


def flip_axis(track_xy, axis_name):
    """
    This function returns a copy of a track with the sign of one axis
    changed, so every x becomes minus x when the axis is x, or every y
    becomes minus y when it is y. This turns a track into its mirror image,
    which is the step that a rotation and a shift cannot do.
    """
    flipped = track_xy.copy()
    column = 0 if axis_name == "x" else 1
    flipped[:, column] = -flipped[:, column]
    return flipped


def fit_rotation_and_shift(moving_xy, target_xy):
    """
    This function finds the one rotation and the one shift that bring the
    moving track as close as possible to the target track, measured over all
    the poses together, using the standard singular value decomposition
    method for this problem. The rotation is forced to be a true rotation
    with no mirror inside it, so any mirror has to be handled before this
    function is called. It returns the moved track and the rotation angle
    in degrees.
    """
    moving_centre = moving_xy.mean(axis=0)
    target_centre = target_xy.mean(axis=0)

    u_matrix, _, vt_matrix = np.linalg.svd((moving_xy - moving_centre).T @ (target_xy - target_centre))
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1] *= -1
        rotation = vt_matrix.T @ u_matrix.T

    moved_xy = (moving_xy - moving_centre) @ rotation.T + target_centre
    angle_degrees = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return moved_xy, angle_degrees


def summarize_gap(moved_xy, target_xy):
    """
    This function measures how far apart the two tracks are at every pose
    after alignment and prints the results in metres. The first number is
    the plain average distance between the two tracks, which is the one to
    quote when someone asks how far apart the satellite and KISS-ICP
    positions are. After it come the root mean square gap, which weighs
    large gaps more heavily, the middle gap, the gap that 95 percent of
    poses stay under, and the largest gap.
    """
    gaps = np.linalg.norm(moved_xy - target_xy, axis=1)
    print(f"  Average distance between KISS-ICP and GNSS: {gaps.mean():.2f} m")
    print(f"  RMSE {np.sqrt((gaps ** 2).mean()):.2f} m | median {np.median(gaps):.2f} m | "
          f"95th percentile {np.percentile(gaps, 95):.2f} m | max {gaps.max():.2f} m")


def save_comparison_picture(kiss_xy, raw_gnss_xy, aligned_gnss_xy, output_path):
    """
    This function draws two side by side panels and saves them as one
    picture. The left panel shows the two tracks exactly as they are in
    their files, each in its own frame, where the mirror is visible. The
    right panel shows the GNSS track after the sign flip, the rotation and
    the shift, drawn on top of the KISS-ICP track, where the two should be
    on top of each other if the fix is right.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    axes[0].plot(kiss_xy[:, 0], kiss_xy[:, 1], label="KISS-ICP", linewidth=1.3)
    axes[0].plot(raw_gnss_xy[:, 0], raw_gnss_xy[:, 1], label="GNSS (as in its file)", linewidth=1.3)
    axes[0].set_title("Before: each track in its own frame")

    axes[1].plot(kiss_xy[:, 0], kiss_xy[:, 1], label="KISS-ICP", linewidth=1.3)
    axes[1].plot(aligned_gnss_xy[:, 0], aligned_gnss_xy[:, 1], label="GNSS (flipped, rotated, shifted)",
                 linewidth=1.3)
    axes[1].set_title("After: GNSS placed on the KISS-ICP track")

    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
        axis.legend()
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    print(f"Saved picture to {output_path}")


if __name__ == "__main__":
    args = parse_command_line_arguments()

    kiss_xy, gnss_xy = load_matching_tracks(args.kiss, args.gnss)
    print(f"Poses used: {len(kiss_xy)}")

    print("\nWithout a flip, rotation and shift only:")
    unflipped_xy, _ = fit_rotation_and_shift(gnss_xy, kiss_xy)
    summarize_gap(unflipped_xy, kiss_xy)

    print(f"\nWith the {args.flip_axis} axis of the GNSS track flipped:")
    flipped_xy = flip_axis(gnss_xy, args.flip_axis)
    aligned_xy, angle_degrees = fit_rotation_and_shift(flipped_xy, kiss_xy)
    summarize_gap(aligned_xy, kiss_xy)
    print(f"  rotation applied after the flip: {angle_degrees:.1f} degrees")

    save_comparison_picture(kiss_xy, gnss_xy, aligned_xy, args.output)






# (venv) shahmeer@ShahmeersLaptop forest-slam-project % python3 "/Users/shahmeer/forest-slam-project/Gault Data/gault_results_all/rosbag2_2026_10_06-20_10_27/align_gnss_to_kiss.py" \
#   --gnss "/Users/shahmeer/forest-slam-project/Gault Data/gault_results_all/rosbag2_2026_10_06-20_10_27/slam_poses_gnss.csv" \
#   --kiss "/Users/shahmeer/forest-slam-project/Gault Data/gault_results_all/rosbag2_2026_10_06-20_10_27/slam_poses_kiss.csv"
# Poses used: 10900

# Without a flip, rotation and shift only:
#   Average distance between KISS-ICP and GNSS: 110.84 m
#   RMSE 125.47 m | median 110.73 m | 95th percentile 218.11 m | max 253.79 m

# With the x axis of the GNSS track flipped:
#   Average distance between KISS-ICP and GNSS: 0.52 m
#   RMSE 0.59 m | median 0.53 m | 95th percentile 0.97 m | max 1.16 m