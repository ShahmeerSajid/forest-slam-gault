# Gault: KISS-ICP poses vs. wheel odometry

## What's in this folder
| File | Contents |
|---|---|
| `rosbag2_2026_10_06-20_10_27_kiss_slam_poses.csv` | KISS-ICP pose, 10,901 rows (1 per LiDAR scan) |
| `rosbag2_2026_10_09-18_37_45_kiss_slam_poses.csv` | KISS-ICP pose, 8,269 rows (1 per LiDAR scan) |
| `trajectory_comparison.png` | KISS-ICP vs wheel odometry (trajectory, per-scan step, cumulative path) |

CSV columns: `timestamp(ns), x, y, z, qx, qy, qz, qw`, same format as the Oxford `slam_poses.csv`.
Each row's timestamp is the bag timestamp of its LiDAR scan, so row i = scan i.
The frame is local: first pose is the identity (0,0,0), so the two bags are NOT in a common frame.

## How they were produced
KISS-ICP (max_range 100 m, min_range 1 m, voxel 1.0 m) run on `/w200_0110/sensors/lidar3d_0/points`
with per-point timestamps for deskewing (`augment_with_kiss_icp.py`), then extracted with
`extract_gault_data.py`.
Baseline = `/w200_0110/platform/odom/filtered` (wheel/EKF odometry), same extraction script.

## Why KISS-ICP is better than the wheel odometry here
There is no ground truth (no GNSS/INS reference was used), so this is evidence, not proof.

1. **Row count / sync.** Wheel odometry is logged at ~50 Hz (41,344 and 54,499 poses), not tied to
   scans. KISS-ICP gives exactly one pose per scan, so poses and clouds pair 1:1 with no interpolation.
2. **Repeatability across the two drives.** Both bags cover the same route. After a rigid alignment,
   the two traverses (matched by arc length) differ by:
   | Source | RMSE | Max |
   |---|---|---|
   | Wheel odometry | 50.8 m | 133 m |
   | KISS-ICP | 7.4 m | 15.6 m |

   KISS-ICP reproduces the same route shape twice. The wheel odometry doesn't (see the PNG: the
   orange shapes differ between bags). This is the strongest evidence, since random wheel/heading
   error would not repeat, while a real route would.
3. **Heading drift.** The wheel trajectory curls away from the KISS-ICP one after the first few hundred
   scans. This is the typical failure of skid-steer wheel odometry in forest: wheel slip on soft or
   uneven ground mostly corrupts yaw, and yaw error grows into large position error.
4. **Elevation.** Wheel odometry has z = 0 for the whole run (2D). KISS-ICP recovers z, with
   ~35 m of elevation change over the route, which the terrain clearly has.
5. **Path length** is similar (KISS 1081-1086 m, wheel 1060-1068 m), so the gap is mostly
   heading, not scale.

## Caveats
- **No ground truth.** These bags have no INS/GNSS reference, so the results above show KISS-ICP is
  more self-consistent than the wheel odometry, not that it is accurate. A new capture with the
  Warthog's INS is planned to validate the KISS-ICP poses against a reference trajectory.
- KISS-ICP is pure odometry, with no loop closure, so it still drifts. The start-to-end gap is
  ~318 m in both bags, which is the route's shape, not a closed loop.
- ~7 m cross-bag RMSE is an upper bound on KISS-ICP's drift plus any real difference in the paths driven.
- The wheel start-to-end gaps (264 m vs 53 m) differ a lot between bags, though the route is nominally
  the same, which again points to wheel error.
- The comparison plot aligns wheel to KISS on the first 300 scans only, so the visible divergence
  includes wheel yaw drift accumulating from there.
- Max per-scan step for KISS is 0.46-0.56 m, with no obvious jumps/failures.










# Summary (added later for clearer understanding):

We drove the robot in Gault Nature Reserve Park along roughly the same forest route twice. Each drive was saved in a separate ROS bag i.e. a recording of the robot’s sensor data.


During each drive, the LiDAR took thousands of scans, like 3D snapshots of the forest.
We then used KISS-ICP to estimate where the robot was and which way it faced for each LiDAR scan. That estimate of the robot's position + orientation  is called a pose.

So each Robot "Pose": its position (x, y, z coords) + orientation (where the robot is facing) during ONE LiDAR Scan (One LIDAR SCAN: thousands of 3D points)

---> First drive: 10,901 scans, each with a pose.
---> Second drive: 8,269 scans, each with a pose.
So we now have what the robot saw, plus an estimated location and direction for EVERY scan. These pose estimates are saved in the CSV files. These CSV files for the Gault Data have the same columns as the Oxford Data's csv files. 


Just like, through the LiDAR Scans we used kiss ICP to estimate the robot's poses,
we use wheel odometry too. Wheel odometry also estimates the robot’s poses: its position and which way it faces, using wheel movement. For wheel odometry, we use wheel sensors. Wheel sensors count wheel rotations. Wheel odometry uses those rotations and the wheel size to estimate how far the robot moved and how much it turned.

- KISS-ICP: LiDAR scans → estimated poses. (i,e estimated robot's x, y, z coordinates and its orientations)
- Wheel odometry: wheel rotations → estimated poses. (i,e estimated robot's x, y, z coordinates and its orientations)


Both the drives (Drive 1 and Drive 2) follow roughly the same real route, thouh not necessarily the exact same tracks. We 
compare Drive 1's kiss icp path with Drive 2's kiss icp path. We also compared Drive 1's wheel odometry path with
Drive 2's wheel odometry path.

What does this mean: Each drive produces a path made from the robot's estimated 
positions, like drawing a line through dots on a map. Because the robot drove roughly the same route twice,
the 2 KISS-ICP paths should have roughly the same shapes. 
Each recording treats its own starting position as (0, 0, 0) and its starting direction as “straight ahead.” 
So the two estimated paths can be shifted or turned relative to each other, even though they describe the same trail.


Forexample, we can imagine drawing the trail on two transparent sheets. We can turn an slide one sheet to make the drawings 
overlap as closely as possible. We cpmapre the points at simolar distances travelled along each path. Forexampel, 500 metres along each route. We then measure the gaps between those points. Those remaining gaps show how much the estimated path shapes differ after removing the difference in starting position and direction.

RMSE summarizes how far apart the two paths are at corresponding points, giving larger gaps more weight than a simple average.
According to the results we get, KISS_ICP paths differed by about 7.40 metres of RMSE. Max difference: 15.6 metres
Wheel-Odometry paths differed by about 50.80 metres of RMSE. Max Difference: 133 metres


What explains this big RMSE difference using the wheel odometry in the Gault forest? 

The 50.8 m RMSE comes from comparing the two wheel-odometry paths.
The robot followed roughly the same forest route twice, but its wheels slipped differently on each drive. Wheel odometry therefore estimated the robot’s turns incorrectly. A small direction error becomes a large position error as the robot travels farther.
So the two estimated paths ended up far apart, giving 50.8 m RMSE as compared to KISS-ICP comparison: 7.4 metres RMSE.





MATCHING A ROBOT POSE TO ITS CORRESPONDING ROBOT SCAN:
Matching each scan to its pose: Each scan is saved as cloud_{timestamp}.pcd, and each pose CSV row includes a timestamp. According to this README, the two new drives have exactly one pose per scan, with matching timestamps. We can therefore match them directly so, no nearest-timestamp search is needed. We should first confirm that both timestamps use the same units.
The earlier 6 GB recording, which I used to develop extract_gault_data.py, had 3,516 scans and 3,504 poses. Those would require timestamp-based matching, potentially using the nearest pose within an acceptable time difference. However, we likely won’t need to process that recording further because we’re now using the two full drives.



