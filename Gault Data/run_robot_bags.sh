#!/usr/bin/env bash
# Extract the 4 robot bags from 2026-10-21 (PCDs + slam_poses.csv), one subfolder per bag. Resumable.
cd "$(dirname "$0")/.."
OUT=/home/williambonilla/gault_out_robot_bags
for bag in /home/williambonilla/robot_bags/rosbag2_2026_10_21-*; do
  .venv/bin/python -I "Gault Data/extract_gault_data.py" --bag-folder "$bag" --output-folder "$OUT/$(basename "$bag")" || exit 1
done
echo ALL DONE
