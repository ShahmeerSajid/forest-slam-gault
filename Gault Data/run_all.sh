#!/usr/bin/env bash
# Extract both Gault bags. Safe to stop (Ctrl-C) and rerun: it resumes where it left off.
cd "$(dirname "$0")/.."
OUT=/home/williambonilla/gault_out
for bag in /home/williambonilla/rosbag2_2026_10_09-18_37_45 /home/williambonilla/Downloads/rosbag2_2026_10_06-20_10_27; do
  .venv/bin/python -I "Gault Data/extract_gault_data.py" --bag-folder "$bag" --output-folder "$OUT/$(basename "$bag")" || exit 1
done
