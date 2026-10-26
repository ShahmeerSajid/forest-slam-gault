"""
LoGG3D-Net Fine-Tuning Launcher
Forest SLAM Project - Oxford Dataset

This script starts one fine-tuning run of LoGG3D-Net on the Oxford sites.
It does not contain the training itself, which is the authors' own
training/train.py inside the cloned repository. What this script does is
put every choice for a run in one visible place, so that whoever presses
the run button, on a laptop or on a lab GPU, runs exactly the settings
that were written down. Before it starts, it saves a small file called
config.json in the folder runs/<run name>, holding the settings and the
reason given for them, and it then runs the authors' training with all of
its output copied into a file called train.log in the same folder. After
the run, those two files show exactly what was run and what happened.

A run needs the repository patched by patch_logg3d_for_oxford.py, a training
pickle made by make_logg3d_training_pickle.py, the authors' weights, and
a GPU machine with the TorchSparse library built with CUDA. The training
code uses the torchpack launcher, which needs a GPU, so a run cannot be
done on a Mac. The check on a Mac is the cpu_smoke_test_logg3d.py script.

How to run it, with the reason written by whoever chose the settings:
    python3 finetune_logg3d.py --run-name wytham_lr1e-4 --hold-out wytham_data --learning-rate 1e-4 --epochs 5 --reason "small rate keeps the pretrained weights"
    python3 finetune_logg3d.py --run-name test --hold-out wytham_data --reason "check the command" --dry-run

With --dry-run it only prints the command and saves config.json, without
training. Each epoch is saved by the training code as its own checkpoint
file in LoGG3D-Net/training/checkpoints, and each one can be scored with
logg3d_baseline.py --checkpoint, only on the held-out site.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys


DEFAULT_CHECKPOINT = "checkpoints/mulran_10cm/2021-09-14_08-59-00_3n24h_MulRan_v10_q29_4s_263039.pth"


def parse_command_line_arguments():
    """
    This function reads the settings for one run. The run name and the
    reason are required, because a run with no name and no stated reason is
    the thing this script exists to prevent. The hold-out site is the one
    left out of training, and it has to match the pickle that was built,
    which by default is looked up from it. The learning rate and the number
    of epochs are the two settings we expect to vary between runs, and the
    other settings are paths and a worker count. Extra settings for the
    authors' training can be added as one quoted string, and they are
    recorded in config.json like everything else.
    """
    parser = argparse.ArgumentParser(description="Start one LoGG3D-Net fine-tuning run on Oxford.")
    parser.add_argument("--run-name", required=True, help="Short name for this run, used for its folder.")
    parser.add_argument("--reason", required=True, help="One sentence on why these settings were chosen.")
    parser.add_argument("--hold-out", default="wytham_data", help="Site left out of training, or 'none'.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Size of each weight update.")
    parser.add_argument("--epochs", type=int, default=5, help="How many passes over the training scans.")
    parser.add_argument("--repo-folder", default="LoGG3D-Net", help="Path to the patched LoGG3D-Net clone.")
    parser.add_argument("--data-root", default="..", help="Folder that holds the four site folders.")
    parser.add_argument("--pickle", default=None, help="Training pickle. Default is looked up from --hold-out.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Starting weights, relative to the repo folder or absolute.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for the run.")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loading workers.")
    parser.add_argument("--extra", default="", help="Extra settings for the authors' training, as one quoted string.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and save config.json only.")
    return parser.parse_args()


def resolve_paths(args):
    """
    This function turns the folder settings into full paths, because the
    authors' training is started from inside the repository folder, where a
    relative path would point somewhere else. The pickle is found from the
    hold-out name when it was not given, using the name that
    make_logg3d_training_pickle.py writes. It stops with a clear message if
    the repository, the pickle or the weights file is missing, so a run does
    not fail only after the GPU machine has been waiting.
    """
    repo_folder = os.path.abspath(args.repo_folder)
    data_root = os.path.abspath(args.data_root)

    hold_out_label = "none" if args.hold_out.lower() == "none" else args.hold_out
    pickle_path = args.pickle or f"oxford_train_holdout_{hold_out_label}.pickle"
    pickle_path = os.path.abspath(pickle_path)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(repo_folder, checkpoint_path)

    for label, path in [("repo folder", repo_folder), ("training pickle", pickle_path),
                        ("starting weights", checkpoint_path), ("data folder", data_root)]:
        if not os.path.exists(path):
            raise SystemExit(f"The {label} was not found: {path}")
    return repo_folder, data_root, pickle_path, checkpoint_path


def build_training_command(args, data_root, pickle_path, checkpoint_path):
    """
    This function builds the exact command that starts the authors'
    training, as a list of words. It tells the training to use the general
    point-pair data loader with our pickle under the name bushwalk, to read
    scans from the data folder, to switch ground-plane removal off, because
    our scans are in the sensor frame and already voxelized, to start from
    the given weights, and to use our learning rate and number of epochs.
    The run name becomes the experiment name, which is part of the
    checkpoint file names.
    """
    command = [
        "torchpack", "dist-run", "-np", str(args.gpus),
        "python", "training/train.py",
        "--train_pipeline", "LOGG3D",
        "--dataset", "GeneralPointSparseTupleDataset",
        "--train_pickles", "{'bushwalk': '" + pickle_path + "'}",
        "--bushwalk_dir", data_root,
        "--gp_rem", "false",
        "--resume_training", "true",
        "--resume_checkpoint", checkpoint_path,
        "--max_epoch", str(args.epochs),
        "--base_learning_rate", str(args.learning_rate),
        "--train_num_workers", str(args.num_workers),
        "--experiment_name", args.run_name,
    ]
    if args.extra.strip():
        command.extend(args.extra.split())
    return command


def save_run_record(run_folder, args, command, pickle_path, checkpoint_path):
    """
    This function writes config.json into the run's folder. It records the
    date, the reason, the hold-out site, the settings, the files used and
    the full command. If a teammate changes anything before the run, the
    change has to be made through this script's settings, so it shows up
    here, and the log's own printed settings can be compared with this file
    afterwards.
    """
    os.makedirs(run_folder, exist_ok=True)
    record = {
        "run_name": args.run_name,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "reason": args.reason,
        "hold_out_site": args.hold_out,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "gpus": args.gpus,
        "pickle": pickle_path,
        "starting_weights": checkpoint_path,
        "extra_settings": args.extra,
        "command": command,
    }
    with open(os.path.join(run_folder, "config.json"), "w") as record_file:
        json.dump(record, record_file, indent=2)


def run_and_log(command, repo_folder, log_path):
    """
    This function runs the training command from inside the repository
    folder and copies everything it prints both to the screen and to the log
    file, line by line as it appears, so that the log is complete even if
    the run is stopped halfway. It returns the exit code of the training.
    """
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(command, cwd=repo_folder, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()
        return process.wait()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    repo_folder, data_root, pickle_path, checkpoint_path = resolve_paths(args)
    command = build_training_command(args, data_root, pickle_path, checkpoint_path)

    run_folder = os.path.join("runs", args.run_name)
    save_run_record(run_folder, args, command, pickle_path, checkpoint_path)
    print("Command:\n  " + " ".join(command))
    print(f"Run record saved to {os.path.join(run_folder, 'config.json')}")

    if args.dry_run:
        print("Dry run, not training.")
    else:
        exit_code = run_and_log(command, repo_folder, os.path.join(run_folder, "train.log"))
        print(f"\nTraining finished with exit code {exit_code}. Log: {os.path.join(run_folder, 'train.log')}")
        print("Checkpoints are in LoGG3D-Net/training/checkpoints (one file per epoch).")
