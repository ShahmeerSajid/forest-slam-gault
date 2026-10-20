# Gault Data Collection Information

## What this document covers

This document describes the INITIAL LiDAR dataset (~149 GB) collected at McGill University's Gault Nature Reserve for the cross-site forest place-recognition project, in which models are trained on the four Oxford forest sites and then tested on Gault.

It covers three drives recorded in October 2026, the pose files that were produced for them, and everything that was checked about those files during analysis in October 2026. Statements that come from William's processing notes and run log are attributed to him. Numbers that were measured by analysing the pose CSV files are given with their values. Anything that has not been verified is labelled as such, because several important questions about this data are still open and are collected in the last section.

Currently these 3 CSV files have no loop closures. tHe goal was to understand pre-processing of the data, and to play around with it initially, so William and I decided to keep things simple. 

Our goal was also to check what was the average distance b/w the robot poses, collected using KISS-ICP (slam_poses_kiss.csv) vs using the Satellite Antenna (slam_poses_gnss.csv)







## Plain-language summary

The robot was driven through the forest along roughly the same route on three different days, and each drive was saved as a separate ROS 2 bag, which is a recording of everything the robot's sensors produced. During each drive the LiDAR took thousands of scans, which are 3D snapshots of the forest made of many thousands of points. A pose is the robot's estimated position (x, y, z) together with the direction it is facing, estimated at the moment of one scan. The scans tell us what the robot saw, and the poses tell us where it was when it saw it, and we need both, because the poses are what allow us to decide which pairs of scans were taken at the same place. Two different methods were used to estimate poses. KISS-ICP estimates them by lining up consecutive LiDAR scans against each other. Wheel odometry estimates them from the rotation of the wheels and the size of the wheels, which tells the robot roughly how far it has moved and how much it has turned. A third source, a satellite positioning (GNSS) track, exists for two of the drives. Root mean square error, written RMSE below, summarises how far apart two paths are at corresponding points and gives larger gaps more weight than a simple average does, so it is always at least as large as the plain average distance.




## The three drives

The three bags are named by the time at which each recording started, and the first timestamp inside every pose file converts to exactly the time in its folder name when read as UTC, so the folder names are UTC times. Drive 1 is rosbag2_2026_10_06-20_10_27, recorded on 6 October 2026 from 20:10:27 to 20:28:37 UTC, a duration of 1,090.0 seconds or 18.2 minutes, with 10,901 LiDAR scans. Drive 2 is rosbag2_2026_10_09-18_37_45, recorded on 9 October 2026 from 18:37:45 to 18:51:32 UTC, a duration of 826.8 seconds or 13.8 minutes, with 8,269 scans. Drive 3 is rosbag2_2026_10_15-20_23_40, recorded on 15 October 2026 from 20:23:40 to 20:35:59 UTC, a duration of 738.6 seconds or 12.3 minutes, with 7,387 scans. Together the three drives hold 26,557 scans. The scans arrive at a steady 0.1 second interval, which is 10 scans per second, and the gaps between consecutive pose timestamps run from 0.095 to 0.105 seconds with a median of 0.100 seconds, so no poses are missing inside any drive, and since the scan counts in the run log equal the pose counts, no scans appear to be missing either.

Each drive folder was produced by William's processing run, whose log ends with the line ALL DONE and contains no errors or warnings. According to that log, every folder holds an individual_clouds folder with the extracted scans as .pcd files, a slam_poses_kiss.csv file with the KISS-ICP poses, and a slam_poses_wheel.csv file with the wheel-odometry poses. The wheel files hold 54,499, 41,344 and 36,931 poses for drives 1, 2 and 3, which is about five times the number of scans, because the wheel odometry is logged at roughly 50 Hz and is not tied to the scans. A satellite pose file called slam_poses_gnss.csv has so far been received for drives 1 and 3 only, and it is not mentioned in the run log. The scan folders themselves have not yet been opened or inspected, so everything said about the scans below rests on the counts in the run log.





## How the poses were produced

According to William's notes, KISS-ICP was run with a maximum range of 100 metres, a minimum range of 1 metre and a voxel size of 1.0 metre, on the LiDAR topic /w200_0110/sensors/lidar3d_0/points, using the per-point timestamps of each scan to correct for the motion of the robot during the scan, which is called deskewing. This was done with a script called augment_with_kiss_icp.py, which adds the KISS-ICP result to the bag, and for drive 3 the run log shows this augmentation step being carried out during the run. The poses and scans were then pulled out of the bags with extract_gault_data.py, the extraction script developed earlier in this project. The wheel-odometry baseline came from the topic /w200_0110/platform/odom/filtered, which is the filtered wheel odometry, extracted with the same script. William has told us not to use the wheel odometry, and the sections below describe it only as the reason why KISS-ICP was chosen.








## File format

Each pose file has the columns timestamp, x, y, z, qx, qy, qz and qw, which are exactly the columns of the Oxford slam_poses.csv files. The timestamp is an integer number of nanoseconds, the position is in metres, and the orientation is a quaternion. In the KISS-ICP files the first pose is the identity, meaning position (0, 0, 0) and no rotation, so each drive is expressed in its own local frame that starts at the robot's starting position and treats the starting direction as straight ahead. Because of that, the KISS-ICP files of different drives are not in a common frame, and the same trail can appear shifted or turned between two drives. William's notes state that each pose file has exactly one row per LiDAR scan, so that row i belongs to scan i.







## The KISS-ICP trajectories

The KISS-ICP file of drive 1 holds 10,901 poses with no missing values. Its path is 1,178.2 metres long in three dimensions and 1,086.1 metres long when height is ignored, the robot ended 318.9 metres from where it started, and its x coordinate runs from −242.4 to 139.3 metres, its y from −25.7 to 428.2 metres and its z from −6.9 to 28.9 metres, a height range of 35.8 metres. The median step between consecutive poses is 0.105 metres and the largest is 0.511 metres, which corresponds to a mean horizontal speed of about 1.0 metres per second, and the largest tilt components of the orientation are 0.324 for qx and 0.111 for qy.

The file of drive 2 holds 8,269 poses, a three-dimensional path of 1,162.3 metres and a horizontal path of 1,080.6 metres, a start-to-end distance of 320.7 metres, an x range from −262.9 to 141.9 metres, a y range from −19.8 to 415.7 metres and a z range from −7.4 to 27.0 metres, a height range of 34.4 metres. Its median step is 0.138 metres, its largest step is 0.565 metres, its mean horizontal speed is about 1.3 metres per second, and its largest tilt components are 0.357 for qx and 0.072 for qy. This file is identical to the drive 2 file that William first sent together with his README.

The file of drive 3 holds 7,387 poses, a three-dimensional path of 1,032.3 metres and a horizontal path of 979.7 metres, a start-to-end distance of 319.5 metres, an x range from −206.2 to 136.6 metres, a y range from −35.4 to 447.8 metres and a z range from −7.5 to 18.4 metres, a height range of 26.0 metres. Its median step is 0.136 metres, its largest step is 0.529 metres, its mean horizontal speed is about 1.3 metres per second, and its largest tilt components are 0.094 for qx and 0.076 for qy. William's notes give the KISS-ICP path length as 1,081 to 1,086 metres, which matches the horizontal path lengths of drives 1 and 2 above, and the three-dimensional lengths are larger because they include the climbs and small movements of the pose estimate. The horizontal path of drive 3 is about 100 metres shorter than those of drives 1 and 2, and this difference has not been explained.

All three drives start and end about 319 to 321 metres apart in a straight line, which is a property of the route and not a closed loop, and none of the files shows jumps or failures, since the largest step in each is below 0.57 metres.








## Why KISS-ICP was chosen over the wheel odometry

William compared the two methods on drives 1 and 2, which cover the same route. After the best rigid alignment, with points matched by distance travelled along each path, the two KISS-ICP traverses differ by an RMSE of 7.4 metres with a largest gap of 15.6 metres, while the two wheel-odometry traverses differ by an RMSE of 50.8 metres with a largest gap of 133 metres. His reasoning is that random wheel error would not repeat from one drive to the next, whereas a real route would, so the fact that KISS-ICP reproduces the same route shape twice and the wheel odometry does not is the strongest evidence that KISS-ICP is the better source. He also points out that the wheel trajectory curls away from the KISS-ICP one after the first few hundred scans, which is the usual failure of skid-steer wheel odometry in forest, where slipping wheels on soft or uneven ground mostly corrupt the heading, and a small heading error turns into a large position error as the robot travels farther. The wheel odometry also has a height of zero for the whole run, while KISS-ICP recovers about 35 metres of elevation change, which the terrain clearly has. The path lengths are similar, with 1,081 to 1,086 metres for KISS-ICP and 1,060 to 1,068 metres for the wheels, so the disagreement comes mostly from heading and not from scale, and the start-to-end gaps of the wheel paths differ strongly between the two bags, 264 metres against 53 metres, although the route is nominally the same. His comparison plot, trajectory_comparison.png, aligns the wheel path to the KISS-ICP path using only the first 300 scans, so the divergence it shows includes the wheel heading error accumulating from that point on. William states that this is evidence and not proof, because there is no ground truth.







## The satellite (GNSS) poses

The satellite pose file of drive 1 holds 10,900 poses and that of drive 3 holds 7,386 poses, which is one fewer than their KISS-ICP files. Their timestamps are identical to the KISS-ICP timestamps, except that the first KISS-ICP pose of each drive has no satellite partner, and their last timestamps are identical too, so every satellite pose can be paired with a KISS-ICP pose and with a scan by timestamp. The positions are in metres in a local frame that starts close to zero, with the first position at (0.19, 0.23, −0.12) for drive 1 and (0.41, 0.33, 0.36) for drive 3, so these files do not contain latitude and longitude. The orientation holds the heading only, because qx and qy are exactly zero in every row, and the first heading is −37.3 degrees for drive 1 and −53.2 degrees for drive 3.

For drive 1 the satellite track is 993.3 metres long in three dimensions and 991.7 metres horizontally, ends 319.3 metres from its start, has a median step of 0.094 metres and a largest step of 0.248 metres, and spans x from −454.6 to 100.5 metres, y from −186.1 to 0.2 metres and z from −4.8 to 2.8 metres, a height range of 7.6 metres. For drive 3 it is 947.4 metres long in three dimensions and 946.7 metres horizontally, ends 320.2 metres from its start, has a median step of 0.128 metres and a largest step of 0.284 metres, and spans x from −453.0 to 95.8 metres, y from −185.8 to 0.3 metres and z from −5.7 to 2.9 metres, a height range of 8.7 metres. Drive 2 has no satellite file.

It is not yet known which sensor and topic these satellite poses come from, whether they are raw receiver output or are filtered or fused with other sensors, or how accurate they are, and William's README states that no GNSS or INS reference was used for these bags, so the origin of the files has to be clarified with him. Earlier project information put the satellite accuracy under the forest canopy at about 5 metres, and the agreement between drives reported below is much tighter than that, so either the accuracy is better than that figure or the errors repeat from drive to drive.







## The satellite tracks are mirror images of the KISS-ICP tracks

When the satellite track and the KISS-ICP track of the same drive are drawn on top of each other, they have the same shape but one is flipped, like a left hand compared with a right hand, and no rotation or shift can turn one into the other. For drive 1, the best rotation and shift without any flip leaves an average distance of 110.84 metres between the two tracks, with an RMSE of 125.47 metres, a median of 110.73 metres, a 95th percentile of 218.11 metres and a largest gap of 253.79 metres. After flipping the sign of one axis of the satellite track and then fitting a rotation and a shift, the average distance falls to 0.52 metres, with an RMSE of 0.59 metres, a median of 0.53 metres, a 95th percentile of 0.97 metres and a largest gap of 1.16 metres. For drive 3 the corresponding numbers are an average of 95.66 metres, an RMSE of 110.17 metres, a median of 93.20 metres, a 95th percentile of 205.39 metres and a largest gap of 229.70 metres without a flip, and an average of 0.29 metres, an RMSE of 0.35 metres, a median of 0.24 metres, a 95th percentile of 0.62 metres and a largest gap of 0.84 metres with a flip.

Three further checks support this conclusion. The first uses only the heading angles and no positions. Over drive 1 the satellite heading turns by −263 degrees while the KISS-ICP heading turns by +274 degrees, and over drive 3 the figures are −278 and +268 degrees, so the two turn in opposite directions. The sum of the two headings stays almost constant, with a standard deviation of 5.5 degrees for drive 1 and 8.9 degrees for drive 3, while their difference swings by 181.2 and 176.3 degrees, which is what a mirror produces. The second check fits five separate stretches of drive 1 on their own. Without a flip the errors are 5.80, 34.45, 23.21, 50.52 and 10.19 metres, and with a flip they are 0.05, 0.17, 0.23, 0.39 and 0.18 metres, for the stretches 0 to 120, 120 to 300, 300 to 600, 600 to 900 and 900 to 1,090 seconds. For drive 3 the errors are 13.12, 15.50, 11.74 and 4.11 metres without a flip and 0.17, 0.15, 0.17 and 0.12 metres with a flip, for the stretches 0 to 120, 120 to 300, 300 to 600 and 600 to 739 seconds. The third check is a control that asks whether the flip is simply too flexible and would make anything fit. If the satellite track is shifted in time by 60 seconds, the best fit with a flip leaves 40.29 metres of error on drive 1 and 52.77 metres on drive 3, and a shift of 300 seconds leaves 112.45 and 76.50 metres, compared with 0.59 and 0.35 metres at the correct timing, so only the correct pairing fits. For the first 30 seconds of drive 1 the two tracks agree even without a flip, with an RMSE of 0.07 metres, because the early path is almost straight, and this is why the mirror only becomes visible after one to two minutes of driving.

The cause of the mirror is not known. A natural explanation is a sign change on one axis in the satellite file, but flipping x and flipping y give exactly the same fit, only turned by half a circle from each other, with rotations of 139.8 and −40.2 degrees for drive 1 and 134.2 and −45.8 degrees for drive 3, so positions alone cannot tell which axis it is. For placing the drives in one frame it makes no difference which of the two is used. The script align_gnss_to_kiss.py applies a chosen flip, fits the rotation and shift, prints these distances and saves a picture of the tracks before and after.





## How well the satellite and KISS-ICP tracks agree

After the flip, the two tracks of the same drive agree to well under a metre on average, which means that they tell the same story about where the robot went, and that the horizontal KISS-ICP path does not drift much sideways on these drives if the satellite poses are accurate. This is agreement and not proof of absolute accuracy. Two differences remain unexplained. The horizontal KISS-ICP path is longer than the satellite path, by 9.5 percent on drive 1 (1,086.1 against 991.7 metres) and by 3.5 percent on drive 3 (979.7 against 946.7 metres), which may be small-scale jitter in the KISS-ICP estimate or a smoother satellite track. The heights disagree strongly, with ranges of 35.8 against 7.6 metres on drive 1 and 26.0 against 8.7 metres on drive 3, and it is not known whether the KISS-ICP height drifts or the satellite height, which is normally the weakest part of satellite positioning, is too flat.







## A shared frame across drives, for the satellite tracks only

The satellite tracks of drives 1 and 3 lie on top of each other exactly as stored, without any alignment. For every fifth pose of drive 3, the distance to the nearest drive 1 pose has a median of 0.5 metres and a mean of 1.0 metres, and 97 percent of these poses lie within 5 metres. Both tracks start at nearly the same place and cover the same extent, with x from about −455 to 100 metres and y from about −186 to 0 metres, so the satellite frame is shared between these two drives even though it starts near zero in each. The KISS-ICP frames, in contrast, are local to each drive. After the best rotation, with points matched by distance travelled, the route shapes of drives 1 and 2 differ by an RMSE of 7.4 metres at a rotation of 2.2 degrees, those of drives 1 and 3 by 21.4 metres at −9.5 degrees, and those of drives 2 and 3 by 16.1 metres at −11.8 degrees, and a mirror flip does not improve the KISS-ICP comparison. This matching by distance travelled is a crude measure, because it assumes that the drives progress along the route at the same pace and it mixes real route differences with drift. The same method applied to the satellite tracks of drives 1 and 3 gives 12.8 metres even though their nearest-point distance has a median of only 0.5 metres, and the nearest-point figure is the one to trust.







## Loop closures

Within each drive, the robot never returns to a place it has already been. Under the project rule of a distance of at most 1.5 metres and at least 50 scans apart, applied after thinning the scans to about 1.1 metres of travel between them, drives 1 and 2 have zero positive pairs, and looser limits of 3, 5 and even 10 metres also give none, because the closest return to an earlier part of the route is about 28.7 metres on drive 1 and 27.6 metres on drive 2 in three dimensions. Using the horizontal position of the raw scans at least 550 scans apart, the closest approach is about 17.9 metres for drive 1 and 9.6 metres for drive 2 using KISS-ICP, 18.1 metres for drive 1 using the satellite track and 43.6 metres for drive 3 using the satellite track, and none of these four tracks has a scan with a partner within 5 metres, while the KISS-ICP track of drive 3 was not checked this way. Each drive is a one-way walk, since the start and end are about 319 to 321 metres apart in a straight line while the path is 980 to 1,090 metres long horizontally.

The three drives do revisit each other's places, so same-place pairs exist across drives. Using the shared satellite frame of drives 1 and 3, and thinning each drive to about 1.1 metres of travel, which leaves 856 keyframes for drive 1 and 811 for drive 3, a radius of 1.5 metres gives 1,590 cross-drive pairs, in which 709 of the 811 drive 3 keyframes (87 percent) and 720 of the 856 drive 1 keyframes (84 percent) have a partner. A radius of 3 metres gives 3,834 pairs with 755 and 771 keyframes having partners, and a radius of 5 metres gives 6,617 pairs with 786 and 804. The median closest approach of a drive 3 keyframe to drive 1 is 0.65 metres, the 90th percentile is 1.68 metres and the largest is 9.16 metres. For comparison, the Oxford sites hold 60, 394, 71 and 1,098 positive pairs, but neighbouring keyframes produce clusters of pairs, so the number of distinct places is smaller than the number of pairs. These pairs have only been counted so far and no Gault label file has been generated yet. Drive 2 cannot join this shared frame directly because it has no satellite file, and it could only be placed through its KISS-ICP track, which is about 7 metres from drive 1, and would need ICP to refine the pairs.







## Matching each scan to its pose

Each scan is saved as a file named cloud_ followed by its timestamp, and each pose file row has a timestamp, and for these three drives there is exactly one KISS-ICP pose for every scan, so row i of the pose file should correspond to scan i in time order, with no nearest-timestamp search required. This still has to be checked on the real scan folders, which have not been opened, in three ways. The number of scan files must equal the number of rows, as the run log says. The timestamp units must be compared, since the pose files use integer nanoseconds whereas in the earlier development sample the scan file names were written as readable UTC date and time strings with microseconds, for example cloud_2026-09-07 05:15:11.677682+00:00.pcd. Finally, the times of a scan and its pose should not be assumed to be identical, because in that sample the first scan and the first pose were 1.4 milliseconds apart. The satellite poses can be paired in the same way, using their timestamps, which coincide with the KISS-ICP timestamps.







## Planned preprocessing of the Gault scans

The Gault scans have to be prepared like the Oxford scans so that results on the two datasets are comparable. Consecutive Gault scans are only 0.105 to 0.138 metres apart, compared with 1.1 to 1.5 metres for Oxford, so the scans will be thinned by distance, keeping one scan for about every 1.1 metres travelled. For the KISS-ICP tracks this leaves about 1,005 scans from drive 1 and 976 from drive 2, which is roughly one scan in 10.8 and one in 8.5, and using the satellite tracks it leaves 856 for drive 1 and 811 for drive 3. The thinned scans will then be voxelized at 0.1 metres, the Oxford setting, although that choice was justified on the Oxford scans by the average distance between original and voxelized points, about 1 centimetre at 0.1 metres against 9 to 10 centimetres at 0.4 metres, and the same test should be repeated on Gault because its sensor is different. Positive pairs will come from different drives, using the 1.5 metre distance rule in the shared satellite frame, because the rule that two scans must be 50 scans apart cannot be applied between separate recordings.






## The earlier development sample

The scan extraction script was developed on a one-minute cut, of about 6 gigabytes, of a first recording made on 7 September 2026, whose full bag is 62.1 gigabytes and holds 160,734 messages over about 6 minutes. That recording used the topic prefix a200_0185 and the LiDAR topic /livox/lidar with 3,516 scans, and an odometry topic /kiss/odometry with 3,504 poses, alongside camera, depth, IMU and platform topics that account for most of the file size. The sample produced 520 scans and 509 poses, with 11 poses missing at 15 scattered places where the odometry skipped a frame, each gap at most 0.4 seconds, over a path of about 14.8 metres in 51.7 seconds, which is a spacing of about 0.03 metres between scans. This recording is not part of the three October drives, whose topics use the prefix w200_0110 and a different LiDAR topic, and William's notes mention a Warthog robot, so the hardware of the October drives may differ from the one used for the sample, although the sensor models are not stated in any of the files received.






## Limitations and open questions

There is no ground truth for any of these poses, so everything above shows how well the sources agree with each other and not how accurate they are. KISS-ICP is odometry with no loop closure, so it still drifts, and the roughly 7 metre difference between drives 1 and 2 is an upper bound on its drift plus any real difference between the paths driven. William plans a new capture with the Warthog's INS to validate the KISS-ICP poses against a reference trajectory. The questions that still need William's answer are the source, accuracy and processing of the satellite poses, including how this fits his statement that no GNSS or INS reference was used; the reason for the mirror and which axis it affects; whether a satellite file exists for drive 2; the sensor hardware of the October drives; and how the cross-drive same-place labels should be built, since each drive has no loop closures of its own. The three drives were recorded on different days, nine days apart for drives 1 and 3, so the scenes may differ through changes in vegetation and ground conditions, which makes cross-drive pairs a real test and not a trivial match. The larger dataset that William is expected to send with loop-closure labels will become the proper Gault test set, and the three drives described here are best used to build and debug the preprocessing and baselines in the meantime.




## How these numbers were produced

All statistics in this document were computed from the pose CSV files alone, using short analysis scripts, and the scan folders of the October drives were not opened. The mirror analysis can be reproduced with align_gnss_to_kiss.py, which takes the KISS-ICP file, the satellite file and the axis to flip. The agreement between satellite tracks of different drives uses the distance from each pose to the nearest pose of the other drive, and the cross-drive pair counts use keyframes thinned to 1.1 metres of horizontal travel and the 1.5, 3 and 5 metre radii stated above.
