"""
Build a pose CSV from the two GNSS receivers (NavSatFix topics) of a bag, one pose per LiDAR scan.

 * position : mean of the two antennas (cancels the antenna offset), in a local ENU frame whose origin is
              the first fix; linearly interpolated to the scan timestamps. Fixes whose two receivers
              disagree wildly or whose covariance is large are dropped before interpolating.
 * heading  : direction of the baseline between the antennas, smoothed, plus a constant mounting offset
              fitted against the direction of travel (course over ground). The fit residual is printed.
 * roll/pitch are not observable -> 0. Output columns match slam_poses.csv (timestamp,x,y,z,qx,qy,qz,qw).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

WGS84_A, WGS84_E2 = 6378137.0, 0.00669437999014


def read_fixes(bag, topic):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    with Reader(Path(bag)) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"{topic} not in bag. NavSatFix topics:\n  " + "\n  ".join(
                sorted(c.topic for c in r.connections if c.msgtype == "sensor_msgs/msg/NavSatFix")))
        for c, t, raw in r.messages(connections=conns):
            m = ts.deserialize_cdr(raw, c.msgtype)
            rows.append((t, m.latitude, m.longitude, m.altitude,
                         max(m.position_covariance[0], m.position_covariance[4]), m.status.status))
    return np.array(rows)


def scan_timestamps(bag, topic):
    with Reader(Path(bag)) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"{topic} not in bag")
        return np.array([t for _, t, _ in r.messages(connections=conns)])


def to_enu(fix, lat0, lon0, h0):
    s = np.sin(np.radians(lat0))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s * s) ** 1.5
    return np.stack([np.radians(fix[:, 2] - lon0) * n * np.cos(np.radians(lat0)),
                     np.radians(fix[:, 1] - lat0) * m, fix[:, 3] - h0], axis=1)


def interp3(t_src, p_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, p_src[:, i]) for i in range(3)], axis=1)


def smooth(x, k):
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def circ_median(a):
    return np.arctan2(np.median(np.sin(a)), np.median(np.cos(a)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-folder", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--lidar-topic", default="/w200_0110/sensors/lidar3d_0/points")
    ap.add_argument("--gps-topic", action="append",
                    default=None, help="NavSatFix topic, give exactly two (default: ins_0/gps_0 and gps_1)")
    ap.add_argument("--max-cov", type=float, default=0.25, help="drop fixes with variance above this (m^2)")
    ap.add_argument("--smooth-s", type=float, default=3.0, help="heading smoothing window (s)")
    a = ap.parse_args()
    topics = a.gps_topic or ["/w200_0110/sensors/ins_0/gps_0/fix", "/w200_0110/sensors/ins_0/gps_1/fix"]
    assert len(topics) == 2, "need exactly two receivers"

    f0, f1 = read_fixes(a.bag_folder, topics[0]), read_fixes(a.bag_folder, topics[1])
    scans = scan_timestamps(a.bag_folder, a.lidar_topic)
    lat0, lon0, h0 = f0[0, 1], f0[0, 2], f0[0, 3]
    p0, p1 = to_enu(f0, lat0, lon0, h0), to_enu(f1, lat0, lon0, h0)
    ok0 = (f0[:, 4] < a.max_cov) & (f0[:, 1] != 0)
    ok1 = (f1[:, 4] < a.max_cov) & (f1[:, 1] != 0)
    print(f"fixes kept: {ok0.sum()}/{len(f0)} and {ok1.sum()}/{len(f1)}")
    t0, t1 = f0[ok0, 0], f1[ok1, 0]
    q0, q1 = p0[ok0], p1[ok1]

    # common time grid = receiver 0 epochs, receiver 1 interpolated onto it
    t = t0
    q1i = interp3(t1, q1, t)
    mid = (q0 + q1i) / 2
    base = q1i - q0
    print(f"antenna baseline: median {np.median(np.linalg.norm(base[:, :2], axis=1)):.3f} m, "
          f"std {np.linalg.norm(base[:, :2], axis=1).std():.3f} m")

    # heading: smoothed baseline direction (unit vectors, so no wrapping issues) + fitted offset vs course
    rate = 1e9 / np.median(np.diff(t))
    k = max(1, int(a.smooth_s * rate))
    bx, by = smooth(base[:, 0], k), smooth(base[:, 1], k)
    base_yaw = np.arctan2(by, bx)
    mx, my = smooth(mid[:, 0], 5), smooth(mid[:, 1], 5)
    vx, vy = np.gradient(mx, t / 1e9), np.gradient(my, t / 1e9)
    speed = np.hypot(vx, vy)
    cog = np.arctan2(vy, vx)
    moving = speed > 0.5
    delta = np.arctan2(np.sin(cog - base_yaw), np.cos(cog - base_yaw))[moving]
    offset = circ_median(delta)
    resid = np.degrees(np.abs(np.arctan2(np.sin(delta - offset), np.cos(delta - offset))))
    print(f"heading offset vs course-over-ground: {np.degrees(offset):.1f} deg; residual median "
          f"{np.median(resid):.1f} deg, 90th pct {np.percentile(resid, 90):.1f} deg (n={moving.sum()})")
    yaw = base_yaw + offset

    # to scan timestamps (only inside the GNSS time span)
    inside = (scans >= t[0]) & (scans <= t[-1])
    ts_ = scans[inside]
    pos = interp3(t, mid, ts_)
    yaw_u = np.unwrap(yaw)
    ys = np.interp(ts_, t, yaw_u)
    df = pd.DataFrame({"timestamp": ts_, "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                       "qx": 0.0, "qy": 0.0, "qz": np.sin(ys / 2), "qw": np.cos(ys / 2)})
    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.parent / "gnss_enu_origin.txt").write_text(
        f"ENU origin (first fix of {topics[0]}): lat {lat0:.9f} lon {lon0:.9f} alt {h0:.3f}\n"
        f"x=east, y=north, z=up (metres). Poses are the mean of the two antennas.\n")
    print(f"{len(df)}/{len(scans)} scans covered -> {out}")
