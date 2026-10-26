#!/usr/bin/env bash
# KISS-ICP poses for the 4 robot bags -> $OUT/<bag>/slam_poses_kiss.csv. Resumable per bag.
# The augmented bag is a full copy (~100 GB), so it is deleted once its pose CSV is saved.
cd "$(dirname "$0")/.."
OUT=/home/williambonilla/gault_out_robot_bags
TMP=/home/williambonilla/augmented_bags
for bag in /home/williambonilla/robot_bags/rosbag2_2026_10_21-*; do
  name=$(basename "$bag")
  [ -f "$OUT/$name/slam_poses_kiss.csv" ] && { echo "$name done, skipping"; continue; }
  rm -rf "$TMP/${name}_augmented.partial"
  .venv/bin/python -I "Gault Data/augment_with_kiss_icp.py" --bag-folder "$bag" --output-folder "$TMP/${name}_augmented" || exit 1
  cp "$TMP/${name}_augmented_kiss_slam_poses.csv" "$OUT/$name/slam_poses_kiss.csv" || exit 1
  rm -rf "$TMP/${name}_augmented"
done
echo ALL DONE
