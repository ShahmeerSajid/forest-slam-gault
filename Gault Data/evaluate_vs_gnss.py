"""
Score pose CSVs (wheel odometry, KISS-ICP) against the dual-GNSS trajectory of one bag.
All trajectories are interpolated to the GNSS CSV timestamps (one per scan) and aligned to GNSS with a
best-fit rigid 3D transform (proper rotation, no scale). A full 3D alignment is required: the KISS-ICP frame is
the LiDAR frame, which may be mounted upside-down, and a 2D yaw fit cannot undo that (it looks like a mirror).
  * ATE xy / 3D : RMSE and max of the position error after alignment
  * RPE         : translation error over 20/50/100 m of travelled distance
  * path ratio  : path length / GNSS path length
GNSS is itself noisy (about 13 cm per fix), so errors below ~0.2 m are at the noise floor.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_trajectories import umeyama  # noqa: E402


def load(path):
    d = pd.read_csv(path)
    yaw = np.unwrap(2 * np.arctan2(d["qz"].to_numpy(), d["qw"].to_numpy()))   # yaw-only or ~yaw-dominant
    full = np.arctan2(2 * (d.qw * d.qz + d.qx * d.qy), 1 - 2 * (d.qy ** 2 + d.qz ** 2)).to_numpy()
    return d["timestamp"].to_numpy(), d[["x", "y", "z"]].to_numpy(), np.unwrap(full)


def resample(t_src, v_src, t_dst):
    v = v_src if v_src.ndim == 2 else v_src[:, None]
    out = np.stack([np.interp(t_dst, t_src, v[:, i]) for i in range(v.shape[1])], axis=1)
    return out if v_src.ndim == 2 else out[:, 0]


def fit_se2(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    ang = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return R, mu_d - R @ mu_s, ang


def rpe(ref, est, window):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    e = []
    for i in range(0, len(ref), 5):
        j = np.searchsorted(s, s[i] + window)
        if j >= len(ref):
            break
        e.append(np.linalg.norm((est[j] - est[i]) - (ref[j] - ref[i])))
    return (np.mean(e), np.percentile(e, 95)) if e else (np.nan, np.nan)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="~/gault_results/<bag> with slam_poses_*.csv")
    ap.add_argument("--plot", default=None, help="write a PNG here")
    a = ap.parse_args()
    folder = Path(a.folder)

    gt, gp, gy = load(folder / "slam_poses_gnss.csv")
    s_ref = np.linalg.norm(np.diff(gp[:, :2], axis=0), axis=1).sum()
    print(f"Reference: dual-GNSS, {len(gt)} poses, path {s_ref:.0f} m, start-end gap "
          f"{np.linalg.norm(gp[-1, :2] - gp[0, :2]):.1f} m\n")

    results, curves = {}, {"GNSS": gp}
    for name, fname in (("wheel", "slam_poses_wheel.csv"), ("KISS-ICP", "slam_poses_kiss.csv")):
        t, p, y = load(folder / fname)
        pe = resample(t, p, gt)
        R, tr = umeyama(pe, gp)
        al = pe @ R.T + tr
        e3 = np.linalg.norm(al - gp, axis=1)
        e2 = np.linalg.norm(al[:, :2] - gp[:, :2], axis=1)
        r20, _ = rpe(gp, al, 20); r50, _ = rpe(gp, al, 50); r100, r100p = rpe(gp, al, 100)
        results[name] = dict(ratio=np.linalg.norm(np.diff(pe[:, :2], axis=0), axis=1).sum() / s_ref,
                             ate2=np.sqrt((e2 ** 2).mean()), ate2max=e2.max(), ate3=np.sqrt((e3 ** 2).mean()),
                             zerr=np.sqrt(((al[:, 2] - gp[:, 2]) ** 2).mean()),
                             r20=r20, r50=r50, r100=r100, r100p=r100p,
                             flip=R[2, 2] < 0)
        curves[name] = al
    print(f"{'metric':42s}" + "".join(f"{k:>12s}" for k in results))
    rows = [("path length / GNSS path", "ratio", "%.3f"), ("ATE xy RMSE (m)", "ate2", "%.2f"),
            ("ATE xy max (m)", "ate2max", "%.2f"), ("ATE 3D RMSE (m)", "ate3", "%.2f"),
            ("z error RMSE (m)", "zerr", "%.2f"), ("RPE 20 m mean (m)", "r20", "%.2f"),
            ("RPE 50 m mean (m)", "r50", "%.2f"), ("RPE 100 m mean (m)", "r100", "%.2f"),
            ("RPE 100 m 95th pct (m)", "r100p", "%.2f"),
            ("sensor frame upside-down (z flipped)", "flip", "%s")]
    for label, key, fmt in rows:
        print(f"{label:42s}" + "".join(f"{fmt % v[key]:>12s}" for v in results.values()))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for k, v in curves.items():
            ax[0].plot(v[:, 0], v[:, 1], label=k, lw=3 if k == "GNSS" else 1)
        ax[0].axis("equal"); ax[0].legend(); ax[0].set_title(f"{folder.name}: trajectories aligned to GNSS (ENU, m)")
        for k, v in curves.items():
            if k != "GNSS":
                ax[1].plot((gt - gt[0]) / 1e9, np.linalg.norm(v[:, :2] - gp[:, :2], axis=1), label=k)
        ax[1].set_yscale("log"); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("horizontal error vs GNSS (m)")
        ax[1].legend(); ax[1].set_title("error vs dual-GNSS (log scale)")
        plt.tight_layout(); plt.savefig(a.plot, dpi=80)
        print("plot:", a.plot)
