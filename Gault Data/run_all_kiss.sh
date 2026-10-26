#!/usr/bin/env bash
# Extract scans + KISS-ICP poses (/kiss/odometry) from the augmented bags. Resumable.
cd "$(dirname "$0")/.."
for name in rosbag2_2026_10_09-18_37_45 rosbag2_2026_10_06-20_10_27; do
  .venv/bin/python -I "Gault Data/extract_gault_data.py" \
    --bag-folder "/home/williambonilla/augmented_bags/${name}_augmented" \
    --output-folder "/home/williambonilla/gault_out_kiss/$name" \
    --odometry-topic /kiss/odometry || exit 1
done
