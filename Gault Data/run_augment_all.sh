#!/usr/bin/env bash
# Create augmented bags (original data + /kiss/odometry). Ctrl-C stops; rerun restarts the current bag.
cd "$(dirname "$0")/.."
OUT=/home/williambonilla/augmented_bags
mkdir -p "$OUT"
for bag in /home/williambonilla/rosbag2_2026_10_09-18_37_45 /home/williambonilla/Downloads/rosbag2_2026_10_06-20_10_27; do
  name=$(basename "$bag")_augmented
  [ -d "$OUT/$name" ] && { echo "$name exists, skipping"; continue; }
  .venv/bin/python -I "Gault Data/augment_with_kiss_icp.py" --bag-folder "$bag" --output-folder "$OUT/$name" || exit 1
done
