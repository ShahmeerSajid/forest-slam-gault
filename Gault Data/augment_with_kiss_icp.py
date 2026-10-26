"""
Make an "augmented" copy of a ROS2 bag: every original message is copied unchanged, plus a
new /kiss/odometry (nav_msgs/Odometry) topic with one KISS-ICP pose per LiDAR scan, stamped
with that scan's bag timestamp. Also saves the poses as kiss_slam_poses.csv next to the bag.

The output is written to <out>.partial and renamed on success, so an interrupted run never
leaves something that looks like a finished bag. KISS-ICP is stateful (local map), so a
stopped run restarts from the beginning (Ctrl-C is handled cleanly).
"""
import argparse
import os
import shutil
import signal
from pathlib import Path

import numpy as np
import pandas as pd
from kiss_icp.config import KISSConfig
from kiss_icp.config.config import DataConfig, MappingConfig
from kiss_icp.kiss_icp import KissICP
from rosbags.rosbag2 import Reader, Writer, StoragePlugin
from rosbags.typesys import Stores, get_typestore

DEFAULT_LIDAR_TOPIC = "/w200_0110/sensors/lidar3d_0/points"
DEFAULT_OUTPUT_TOPIC = "/kiss/odometry"
POINTFIELD_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print("\nStop requested, closing up (partial output will be discarded)...")


def decode_cloud(msg):
    """PointCloud2 -> (xyz float64 [N,3], per-point time normalised to [0,1] or zeros)."""
    fields = {f.name: f for f in msg.fields}
    end = ">" if msg.is_bigendian else "<"
    names = ["x", "y", "z"] + (["t"] if "t" in fields else [])
    dtype = np.dtype({
        "names": names,
        "formats": [end + POINTFIELD_DTYPES[fields[n].datatype] for n in names],
        "offsets": [fields[n].offset for n in names],
        "itemsize": msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    if "t" in names:
        t = raw["t"].astype(np.float64)
        span = t.max() - t.min()
        t = (t - t.min()) / span if span > 0 else np.zeros_like(t)
    else:
        t = np.zeros(len(xyz))
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], t[ok]


def matrix_to_quaternion(R):
    """3x3 rotation -> (qx, qy, qz, qw)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def make_odometry(typestore, timestamp, frame_id, pose):
    T = typestore.types
    qx, qy, qz, qw = matrix_to_quaternion(pose[:3, :3])
    stamp = T["builtin_interfaces/msg/Time"](sec=timestamp // 10**9, nanosec=timestamp % 10**9)
    header = T["std_msgs/msg/Header"](stamp=stamp, frame_id="kiss_odom")
    p = pose[:3, 3]
    ppose = T["geometry_msgs/msg/Pose"](
        position=T["geometry_msgs/msg/Point"](x=p[0], y=p[1], z=p[2]),
        orientation=T["geometry_msgs/msg/Quaternion"](x=qx, y=qy, z=qz, w=qw))
    zeros = np.zeros(36)
    twist = T["geometry_msgs/msg/Twist"](
        linear=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
        angular=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0))
    return T["nav_msgs/msg/Odometry"](
        header=header, child_frame_id=frame_id,
        pose=T["geometry_msgs/msg/PoseWithCovariance"](pose=ppose, covariance=zeros),
        twist=T["geometry_msgs/msg/TwistWithCovariance"](twist=twist, covariance=zeros))


def augment(bag_folder, out_folder, lidar_topic, odom_topic, max_range, min_range, voxel_size):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    partial = Path(str(out_folder) + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    kiss = KissICP(KISSConfig(data=DataConfig(max_range=max_range, min_range=min_range),
                              mapping=MappingConfig(voxel_size=voxel_size)))
    rows = []

    with Reader(Path(bag_folder)) as reader, \
            Writer(partial, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        if not any(c.topic == lidar_topic for c in reader.connections):
            raise SystemExit(f"{lidar_topic} not in bag. Topics:\n  "
                             + "\n  ".join(sorted(c.topic for c in reader.connections)))
        mapping = {}
        for c in reader.connections:
            mapping[c.id] = writer.add_connection(
                c.topic, c.msgtype, msgdef=c.msgdef.data, rihs01=c.digest or "RIHS01_" + "0" * 64,  # recovered bags may have no type hashes
                serialization_format=c.ext.serialization_format,
                offered_qos_profiles=c.ext.offered_qos_profiles)
        odom_conn = writer.add_connection(odom_topic, "nav_msgs/msg/Odometry", typestore=typestore)
        total = next(c.msgcount for c in reader.connections if c.topic == lidar_topic)

        for conn, timestamp, raw in reader.messages():
            writer.write(mapping[conn.id], timestamp, raw)
            if conn.topic != lidar_topic:
                continue
            xyz, t = decode_cloud(typestore.deserialize_cdr(raw, conn.msgtype))
            kiss.register_frame(xyz, t)
            pose = kiss.last_pose
            writer.write(odom_conn, timestamp,
                         typestore.serialize_cdr(make_odometry(typestore, timestamp, "lidar", pose),
                                                 "nav_msgs/msg/Odometry"))
            qx, qy, qz, qw = matrix_to_quaternion(pose[:3, :3])
            rows.append({"timestamp": timestamp, "x": pose[0, 3], "y": pose[1, 3], "z": pose[2, 3],
                         "qx": qx, "qy": qy, "qz": qz, "qw": qw})
            if len(rows) % 100 == 0 or len(rows) == total:
                print(f"KISS-ICP {len(rows)}/{total} scans", flush=True)
            if stop_requested:
                break

    if stop_requested:
        shutil.rmtree(partial, ignore_errors=True)
        raise SystemExit("Stopped; partial output removed. Rerun to start over.")
    os.replace(partial, out_folder)
    csv_path = Path(str(out_folder) + "_kiss_slam_poses.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Augmented bag: {out_folder}\nPoses: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Copy a ROS2 bag and add a KISS-ICP odometry topic.")
    ap.add_argument("--bag-folder", required=True)
    ap.add_argument("--output-folder", required=True, help="Augmented bag folder to create")
    ap.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    ap.add_argument("--odometry-topic", default=DEFAULT_OUTPUT_TOPIC, help="Topic to add (default: %(default)s)")
    ap.add_argument("--max-range", type=float, default=100.0)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--voxel-size", type=float, default=1.0, help="KISS-ICP map voxel size in metres")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if Path(a.output_folder).exists():
        raise SystemExit(f"{a.output_folder} already exists; refusing to overwrite.")
    augment(a.bag_folder, a.output_folder, a.lidar_topic, a.odometry_topic,
            a.max_range, a.min_range, a.voxel_size)
