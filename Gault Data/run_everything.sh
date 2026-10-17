#!/usr/bin/env bash
# For every bag: scans (shared) + poses without KISS-ICP (wheel odom) + poses with KISS-ICP.
# Output: ~/gault_results/<bag name>/{individual_clouds/, slam_poses_wheel.csv, slam_poses_kiss.csv}
# Resumable: rerun to continue. The GNSS CSV for the newest bag is made by gnss_poses.py.
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
RES=/home/williambonilla/gault_results
AUG=/home/williambonilla/augmented_bags
declare -A BAG=(
  [rosbag2_2026_10_06-20_10_27]=/home/williambonilla/Downloads/rosbag2_2026_10_06-20_10_27
  [rosbag2_2026_10_09-18_37_45]=/home/williambonilla/rosbag2_2026_10_09-18_37_45
  [rosbag2_2026_10_15-20_23_40]=/home/williambonilla/rosbag2_2026_10_15-20_23_40
)
for name in rosbag2_2026_10_06-20_10_27 rosbag2_2026_10_09-18_37_45 rosbag2_2026_10_15-20_23_40; do
  echo "=== $name: scans + wheel-odometry poses (without KISS-ICP)"
  $PY "Gault Data/extract_gault_data.py" --bag-folder "${BAG[$name]}" --output-folder "$RES/$name" \
      --poses-csv-name slam_poses_wheel.csv || exit 1
  if [ ! -d "$AUG/${name}_augmented" ]; then
    echo "=== $name: adding KISS-ICP"
    $PY "Gault Data/augment_with_kiss_icp.py" --bag-folder "${BAG[$name]}" --output-folder "$AUG/${name}_augmented" || exit 1
  fi
  echo "=== $name: KISS-ICP poses"
  $PY "Gault Data/extract_gault_data.py" --bag-folder "$AUG/${name}_augmented" --output-folder "$RES/$name" \
      --odometry-topic /kiss/odometry --poses-csv-name slam_poses_kiss.csv --skip-lidar || exit 1
done
echo ALL DONE
