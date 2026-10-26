"""
LoGG3D-Net Training Smoke Test on the CPU
Forest SLAM Project - Oxford Dataset

This script runs a handful of real training steps of LoGG3D-Net on the
CPU, so that the whole fine-tuning chain can be checked on a laptop before
anyone books GPU time. It uses the authors' own data loader, model and
loss functions, with the same settings as the real run, but it replaces
their torchpack launcher and the GPU with a plain loop. For each step it
prints how many scans were in the batch, how large the sparse tensor is,
the scene loss, the point loss and the total loss, so that you can see what
one batch is and how the loss behaves. It saves nothing, and a few steps
will not make the weights better, so the numbers show that the chain works
and nothing about the final quality.

The thing to look for is that every step runs without an error and that the
loss is a normal number, not nan or infinity. A run on the CPU is slow, so
ten steps is plenty. If a step prints "skipped", the data loader found no
matching points between the query scan and its positive, which the real
training skips in the same way.

The repository must be patched by patch_logg3d_for_oxford.py and TorchSparse
must be installed, and the weights are loaded the same way as in the real
run. The settings are not read from the command line of the authors' code,
so the options of this script are the ones that matter.

How to run it:
    python3 cpu_smoke_test_logg3d.py --pickle oxford_train_holdout_wytham_data.pickle --data-root ..
"""

import argparse
import os
import sys

import torch


def parse_command_line_arguments():
    """
    This function reads the settings for the test. The pickle and the data
    folder are the same ones the real run uses. The steps setting is the
    number of training steps to run, the learning rate is the same
    setting as in a real run, and the checkpoint is the starting weights. The
    weights are optional here, since a test of the chain does not need them,
    but loading them also tests that the checkpoint fits the network.
    """
    parser = argparse.ArgumentParser(description="A few LoGG3D-Net training steps on the CPU.")
    parser.add_argument("--pickle", required=True, help="Training pickle made by make_logg3d_training_pickle.py.")
    parser.add_argument("--data-root", default="..", help="Folder that holds the four site folders.")
    parser.add_argument("--repo-folder", default="LoGG3D-Net", help="Path to the patched LoGG3D-Net clone.")
    parser.add_argument("--checkpoint", default=None, help="Starting weights file, optional.")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps to run.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Size of each weight update.")
    return parser.parse_args()


def load_authors_code(args):
    """
    This function sets up the authors' code for use from this script. It puts
    the repository on Python's search path and fills in the command-line
    settings that their config module reads, using the same settings as the
    real run, with no data loading workers, because a single process is
    easier to debug. It then imports their config, data loader factory,
    model factory and loss helpers, and returns them together with the
    finished config.
    """
    repo_folder = os.path.abspath(args.repo_folder)
    sys.path.insert(0, repo_folder)
    sys.argv = [
        "smoke_test",
        "--dataset", "GeneralPointSparseTupleDataset",
        "--train_pickles", "{'bushwalk': '" + os.path.abspath(args.pickle) + "'}",
        "--bushwalk_dir", os.path.abspath(args.data_root),
        "--gp_rem", "false",
        "--train_num_workers", "0",
        "--base_learning_rate", str(args.learning_rate),
    ]
    from config.train_config import get_config
    from utils.data_loaders.make_dataloader import make_data_loader
    from models.pipeline_factory import get_pipeline
    from training import train_utils

    config = get_config()
    return config, make_data_loader, get_pipeline, train_utils


def load_weights_if_given(model, checkpoint_path):
    """
    This function loads the authors' weights into the model when a file was
    given, and prints a note saying so. If none was given, the test runs from
    the random starting weights, which is enough to test the chain but means
    the loss values will not be typical of a fine-tuning run.
    """
    if checkpoint_path is None:
        print("No checkpoint given: running from random weights.")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded starting weights from {checkpoint_path}")


def run_steps(model, loader, config, train_utils, steps):
    """
    This function runs the training steps. Each step takes one batch from the
    loader, which is one query scan together with its same-place scans and
    its different-place scans, runs the network on all of them at once, and
    computes the scene loss, which compares the descriptors, and the point
    loss, which compares the local features of matching points. It then
    adds the two losses, lets the optimizer update the weights, and prints
    the size of the batch and the three loss values. This follows the loop in
    the authors' training/train.py, step by step.
    """
    loss_function = train_utils.get_loss_function(config)
    point_loss_function = train_utils.get_point_loss_function(config)
    optimizer = train_utils.get_optimizer(config, model.parameters())

    model.train()
    iterator = iter(loader)
    for step in range(steps):
        batch = next(iterator)
        batch_st = batch[0]
        pos_pairs = batch[1]["pos_pairs"]

        if not pos_pairs.ndim == 2:
            print(f"step {step}: skipped, no matching points between the query and its positive")
            continue

        scans_in_batch = len(torch.unique(batch_st.C[:, -1]))
        output = model(batch_st)
        scene_loss = loss_function(output[0], config)
        point_loss = point_loss_function(output[1][0], output[1][1], pos_pairs, config)
        loss = config.scene_loss_weight * scene_loss + config.point_loss_weight * point_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step}: scans in batch {scans_in_batch} | points {batch_st.F.shape[0]} | "
              f"scene loss {scene_loss.item():.4f} | point loss {point_loss.item():.4f} | "
              f"total {loss.item():.4f} | matching point pairs {pos_pairs.shape[0]}")


if __name__ == "__main__":
    args = parse_command_line_arguments()
    config, make_data_loader, get_pipeline, train_utils = load_authors_code(args)

    print(f"Batch size in the config: {config.batch_size} (one query tuple per step)")
    print(f"Positives per query: {config.positives_per_query}, negatives per query: {config.negatives_per_query}")

    loader = make_data_loader(config, config.train_phase, config.batch_size,
                              num_workers=config.train_num_workers, shuffle=True)
    print(f"Scans in the training pickle: {len(loader.dataset)}")

    model = get_pipeline(config.train_pipeline)
    load_weights_if_given(model, args.checkpoint)
    model = model.to("cpu")

    run_steps(model, loader, config, train_utils, args.steps)
    print("\nSmoke test finished without errors.")
