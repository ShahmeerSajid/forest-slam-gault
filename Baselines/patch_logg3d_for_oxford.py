"""
LoGG3D-Net Patch for Custom Training
Forest SLAM Project - Oxford Dataset

The LoGG3D-Net authors added a "general" data loader for custom datasets,
but it is not fully connected in the released code. The training script
only knows the KITTI and MulRan loaders, and the general loader asks for
settings (bushwalk_dir, ugv_dir, apollo_dir, downsample) that the training
config does not define, so training on our own data stops with an error
before it starts. This script makes six small edits inside the cloned
LoGG3D-Net folder to connect the pieces. It adds the missing settings to
config/train_config.py, registers the general loaders in
utils/data_loaders/make_dataloader.py, and changes how training/train.py
starts from a checkpoint, so that it loads only the saved weights, begins
at epoch 0 with a fresh optimizer, and can read a checkpoint that has no
optimizer inside. The last one is what fine-tuning needs, since the
released checkpoints may hold only weights. Two more edits fix problems
found by reading the training code. The command line cannot pass the
train_pickles setting as released, because it is declared as a plain dict
type, which cannot be built from text, so it is changed to read a Python
dictionary written as text. And training saves every epoch to the same file
name, so each epoch overwrites the one before, and the edit adds the epoch
number to the file name so every epoch is kept.

Each edited file is first copied to a .orig file next to it, and running
the script twice is safe, because an edit that is already in place is
skipped. The edits were checked against the repository version that has
the commit message "Added dropbox link for checkpoints." If a file does not
look as expected, the script stops and says which one, instead of editing
blindly.

How to run it:
    python3 patch_logg3d_for_oxford.py --repo-folder LoGG3D-Net
"""

import argparse
import os
import shutil


def parse_command_line_arguments():
    """
    This function reads the one setting this script needs, which is the path
    to the cloned LoGG3D-Net folder.
    """
    parser = argparse.ArgumentParser(description="Patch a LoGG3D-Net clone for training on Oxford data.")
    parser.add_argument("--repo-folder", default="LoGG3D-Net", help="Path to the cloned LoGG3D-Net folder.")
    return parser.parse_args()


def apply_edit(file_path, old_text, new_text, marker):
    """
    This function replaces one exact piece of text in a file with new text.
    It first looks for a marker that only exists after the edit, and if the
    marker is already there it does nothing, which makes the patch safe to
    run twice. Before the first edit of a file it saves a .orig copy. If
    the old text is not found exactly once, it stops with a clear message,
    because that means the file is not the version this patch was written
    for.
    """
    with open(file_path, "r") as file:
        content = file.read()

    if marker in content:
        print(f"  already patched: {file_path}")
        return

    if content.count(old_text) != 1:
        raise SystemExit(f"Cannot patch {file_path}: the expected text was found "
                         f"{content.count(old_text)} times instead of once.")

    backup_path = file_path + ".orig"
    if not os.path.exists(backup_path):
        shutil.copyfile(file_path, backup_path)

    with open(file_path, "w") as file:
        file.write(content.replace(old_text, new_text))
    print(f"  patched: {file_path}")


def patch_train_config(repo_folder):
    """
    This function adds the settings the general loader reads but the config
    does not define. The bushwalk_dir setting is the important one, since it
    is the folder that holds our four site folders. The ugv_dir and
    apollo_dir settings are unused by us but are read when the loader
    starts, so they only need to exist. The downsample setting is also read
    by the loader and is set to false because our scans are already
    voxelized. The bushwalk entry is added to the ground-plane values for
    completeness, although we switch ground removal off when we train.
    """
    file_path = os.path.join(repo_folder, "config", "train_config.py")
    old_text = "data_arg.add_argument('--mulran_dir', type=str,"
    new_text = (
        "data_arg.add_argument('--bushwalk_dir', type=str, default='')  # OXFORD_PATCH\n"
        "data_arg.add_argument('--ugv_dir', type=str, default='')\n"
        "data_arg.add_argument('--apollo_dir', type=str, default='')\n"
        "data_arg.add_argument('--downsample', type=str2bool, default=False)\n"
        "data_arg.add_argument('--mulran_dir', type=str,"
    )
    apply_edit(file_path, old_text, new_text, "OXFORD_PATCH")

    old_gp = "'apollo': 1.6, 'kitti':1.5, 'mulran':0.9"
    new_gp = "'apollo': 1.6, 'kitti':1.5, 'mulran':0.9, 'bushwalk': 1.5"
    apply_edit(file_path, old_gp, new_gp, "'bushwalk': 1.5")


def patch_dataloader_factory(repo_folder):
    """
    This function makes the training script able to find the general
    loaders by name. It imports them and adds them to the list of known
    dataset classes, so that passing --dataset GeneralPointSparseTupleDataset
    on the command line works.
    """
    file_path = os.path.join(repo_folder, "utils", "data_loaders", "make_dataloader.py")
    old_import = "from utils.data_loaders.kitti.kitti_dataset import *\n"
    new_import = (
        "from utils.data_loaders.kitti.kitti_dataset import *\n"
        "from utils.data_loaders.general.general_sparse_dataset import *  # OXFORD_PATCH\n"
    )
    apply_edit(file_path, old_import, new_import, "OXFORD_PATCH")

    old_list = "    MulRanDataset, MulRanTupleDataset, MulRanSparseTupleDataset, MulRanPointSparseTupleDataset\n]"
    new_list = (
        "    MulRanDataset, MulRanTupleDataset, MulRanSparseTupleDataset, MulRanPointSparseTupleDataset,\n"
        "    GeneralTupleDataset, GeneralSparseTupleDataset, GeneralPointSparseTupleDataset\n]"
    )
    apply_edit(file_path, old_list, new_list, "GeneralPointSparseTupleDataset\n]")


def patch_resume_logic(repo_folder):
    """
    This function changes how training starts from a checkpoint. The
    original code insists on finding the epoch number and the optimizer
    state inside the file, and it carries on counting from that epoch. For
    fine-tuning we want to take only the learned weights, start counting
    again from epoch 0, and use a new optimizer with our own small learning
    rate. The edit also loads the file onto the CPU first and allows the
    full file format, so it works with the authors' checkpoints on newer
    PyTorch versions.
    """
    file_path = os.path.join(repo_folder, "training", "train.py")
    old_text = (
        "        checkpoint = torch.load(save_path)\n"
        "        starting_epoch = checkpoint['epoch']\n"
        "        model.load_state_dict(checkpoint['model_state_dict'])\n"
        "        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n"
    )
    new_text = (
        "        checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)  # OXFORD_PATCH\n"
        "        starting_epoch = 0\n"
        "        model.load_state_dict(checkpoint['model_state_dict'])\n"
    )
    apply_edit(file_path, old_text, new_text, "OXFORD_PATCH")


def patch_pickle_setting_and_checkpoint_names(repo_folder):
    """
    This function makes two more edits. In the config, the train_pickles
    setting is changed from the plain dict type, which fails on any text
    given on the command line, to a reader for Python dictionaries written
    as text, so that --train_pickles "{'bushwalk': '/path/file.pickle'}"
    works. In training/train.py, the saved checkpoint name gets the epoch
    number added, so that each epoch is written to its own file and we can
    later score every epoch and find the one where the held-out score is
    best.
    """
    config_path = os.path.join(repo_folder, "config", "train_config.py")
    old_type = "data_arg.add_argument('--train_pickles', type=dict, default={"
    new_type = "data_arg.add_argument('--train_pickles', type=__import__('ast').literal_eval, default={"
    apply_edit(config_path, old_type, new_type, "__import__('ast').literal_eval")

    train_path = os.path.join(repo_folder, "training", "train.py")
    old_name = "            save_path = str(save_path) + '/' + cfg.experiment_name\n"
    new_name = "            save_path = str(save_path) + '/' + cfg.experiment_name + '_epoch' + str(epoch)  # OXFORD_PATCH\n"
    apply_edit(train_path, old_name, new_name, "'_epoch' + str(epoch)")


if __name__ == "__main__":
    args = parse_command_line_arguments()
    if not os.path.isdir(os.path.join(args.repo_folder, "training")):
        raise SystemExit(f"{args.repo_folder} does not look like the LoGG3D-Net folder.")

    print("Patching config...")
    patch_train_config(args.repo_folder)
    print("Patching data loader list...")
    patch_dataloader_factory(args.repo_folder)
    print("Patching checkpoint loading...")
    patch_resume_logic(args.repo_folder)
    print("Patching pickle setting and checkpoint names...")
    patch_pickle_setting_and_checkpoint_names(args.repo_folder)
    print("\nDone. Originals are saved next to each file as .orig")
