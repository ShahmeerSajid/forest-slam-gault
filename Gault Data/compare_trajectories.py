"""
Compare trajectories from a bag: the KISS-ICP poses (from the augmented bag's /kiss/odometry)
against other odometry topics in the original bag (wheel/EKF, Fixposition INS). All are
interpolated to the LiDAR scan timestamps, SE(3)-aligned (no scale) to the reference, and
scored with ATE RMSE and relative translation error over fixed path-length windows.
"""
import argparse
from pathlib import Path
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def read_positions(bag, topic):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    t, p = [], []
    with Reader(Path(bag)) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            return None
        for c, stamp, raw in r.messages(connections=conns):
            pos = ts.deserialize_cdr(raw, c.msgtype).pose.pose.position
            t.append(stamp); p.append((pos.x, pos.y, pos.z))
    return (np.array(t), np.array(p)) if t else None


def interp(t_src, p_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, p_src[:, i]) for i in range(3)], axis=1)


def umeyama(src, dst):
    """Rigid transform (R, t) minimising |R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def rpe(ref, est, window_m):
    """Mean/max translation error between segments of `window_m` metres of reference path."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    errs = []
    for i in range(0, len(ref), 5):
        j = np.searchsorted(s, s[i] + window_m)
        if j >= len(ref):
            break
        errs.append(np.linalg.norm((est[j] - est[i]) - (ref[j] - ref[i])))
    return (np.mean(errs), np.max(errs)) if errs else (np.nan, np.nan)


def path_len(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-folder", required=True, help="Original bag (has the other odometry topics)")
    ap.add_argument("--augmented-folder", required=True, help="Augmented bag with /kiss/odometry")
    ap.add_argument("--kiss-topic", default="/kiss/odometry")
    ap.add_argument("--compare-topic", action="append", required=True,
                    help="Odometry topic(s) to compare against KISS-ICP (repeatable)")
    a = ap.parse_args()

    kiss_t, kiss_p = read_positions(a.augmented_folder, a.kiss_topic)
    print(f"KISS-ICP: {len(kiss_t)} poses, path {path_len(kiss_p):.0f} m, "
          f"start-end gap {np.linalg.norm(kiss_p[-1] - kiss_p[0]):.1f} m")
    for topic in a.compare_topic:
        res = read_positions(a.bag_folder, topic)
        print(f"\n== {topic}")
        if res is None:
            print("  no messages"); continue
        t, p = res
        ok = (kiss_t >= t[0]) & (kiss_t <= t[-1])
        if ok.sum() < 10:
            print("  no time overlap with scans"); continue
        other = interp(t, p, kiss_t[ok]); kiss = kiss_p[ok]
        if not np.isfinite(other).all() or np.ptp(other, axis=0).max() < 1:
            print("  trajectory is empty/constant, skipping"); continue
        R, tr = umeyama(other, kiss)
        aligned = other @ R.T + tr
        err = np.linalg.norm(aligned - kiss, axis=1)
        print(f"  {ok.sum()} scans compared; path {path_len(other):.0f} m "
              f"(KISS {path_len(kiss):.0f} m, ratio {path_len(other)/path_len(kiss):.3f}); "
              f"start-end gap {np.linalg.norm(other[-1]-other[0]):.1f} m")
        print(f"  ATE after SE3 alignment: RMSE {np.sqrt((err**2).mean()):.2f} m, max {err.max():.2f} m")
        for w in (20, 50, 100):
            m, mx = rpe(kiss, aligned, w)
            print(f"  relative error over {w:>3} m: mean {m:.2f} m, max {mx:.2f} m")
