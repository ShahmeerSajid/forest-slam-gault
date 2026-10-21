"""
LoGG3D-Net Baseline Script


This is our third baseline, to sit next to Scan Context and ICP. Scan
Context and ICP are classical methods that need no training. 

LoGG3D-Net is developed by NoRA Lab and it is 
a learned method, a neural network from the ICRA 2022 paper 

"LoGG3D-Net:
Locally Guided Global Descriptor Learning for 3D Place Recognition", 

LoGG3D-Net uses a sparse 3D CNN.
- A normal image CNN processes pixels.
- This network processes 3D LiDAR points arranged into occupied voxels—small cubes in space.
- It combines the features it finds into one descriptor representing the scan’s place.
“Sparse” means it focuses on occupied cubes instead of processing all the empty space.

A sparse 3D CNN is a neural network that finds patterns in 3D data, such as my LiDAR scans.
- 3D: It processes shapes in space, rather than pixels in a flat image.
- CNN: It learns local patterns, such as surfaces and edges.
- Sparse: It processes occupied voxels—small cubes containing scan points—while skipping empty space to save computation.
In LoGG3D-Net, these learned features are combined into a descriptor representing the place.

The purpose of this code is to find out how well an existing learned method, 
used as it comes, does on our forest scans. We do not train it. We load the 
weights the authors published, run every scan of every labeled pair through
the network, and call a pair the same place when its two descriptors are close. 
The score is the cosine distance between the two descriptors, where a low number 
means the same place, exactly as in the Scan Context script. We then grade it
with the same AUC and best F1 measures on the same labeled pairs, so all
three baselines can be compared fairly.

What you need before running it:
1. An NVIDIA GPU with CUDA. LoGG3D-Net is built on TorchSparse, which does
   not run on a Mac or on a computer without a GPU, so this script has to
   run on William's GPU machine or another Linux GPU machine.
2. The authors' code, cloned next to this script as a folder called
   LoGG3D-Net, from https://github.com/csiro-robotics/LoGG3D-Net, with its
   environment installed as described in its README (PyTorch, TorchSparse
   1.4.0, Open3D, torchpack and mpi4py).
3. The authors' pretrained checkpoints, downloaded from the links in their
   README and extracted into LoGG3D-Net/checkpoints. Only load checkpoints
   from the authors, because loading one runs the pickle format's code.
4. Our voxelized Oxford scans and label files in the same folders the other
   scripts use. Do not add the LoGG3D-Net folder or its checkpoints to git.

How to run it, from the project folder:
    python3 logg3d_baseline.py --max-pairs 40
    python3 logg3d_baseline.py
    python3 logg3d_baseline.py --checkpoint checkpoints/kitti_10cm_loo/<file>.pth

]

Two things to keep in mind when reading the results. First, the published
checkpoints were trained on city driving data (KITTI and MulRan), so weaker
results in forests are expected, and that is a real finding about how an
existing learned method carries over to forests. Second, the network expects
four numbers per point, the position x, y, z and the LiDAR return intensity,
and our voxelized scans keep only the position. We therefore give every
point the same constant intensity, zero by default, which carries no
information. This is a known mismatch with how the network was trained and
may lower its scores a little.







Imagine we want to ask: “Do these two scans show the same place?”

The goal is simple: give the model two forest scans and check whether it recognizes the same place.
Each function has one job:
1. parse_command_line_arguments() — reads the settings.
   For example: “Use this checkpoint” or “Test only 40 pairs.” 
2. load_model() — gets the model ready.
   First, it builds the neural network. Then it loads the authors’ saved weights into it. The model is now ready to examine scans using what it already learned.
3. compute_descriptor() — gives one scan to the model.
   It opens a scan file containing the LiDAR points and prepares those points for the model. The model turns the scan into a short list of numbers describing the place. This list is called a descriptor. It keeps the result in memory to reuse later.
4. cosine_distance() — compares two lists of numbers.
   Once both scans have descriptors, this function compares them. A small difference suggests the same place; a large difference suggests different places.
5. score_pairs_for_site() — repeats this for one forest.
   It reads your CSV, which says which two scans to compare and whether they really show the same place. For each pair, it gets both descriptors and compares them. It keeps the comparison score alongside the correct answer.
6. The main block — runs the whole process.
   It loads the model, tests each forest, and calculates the performance scores using the correct answers.
For example, for scan A and scan B:
- Give scan A to the model → get its numbers.
- Give scan B to the model → get its numbers.
- Compare those numbers → get a similarity score.
- Check your label → did the model get it right?
Throughout this process, the weights stay unchanged. The model is being tested, not trained.

The model uses the Logg3D-Nets weightsto turn scan A’s points into a descriptor — a list of numbers describing that scan.
Scan points + model with pretrained weights → descriptor
The weights stay the same for every scan, but each scan produces its own descriptor because its points are different.


Important to know:
1) What do the LoGG3D weights do?

The network uses those learned weights to turn each scan into a descriptor. The weights were
trained so that scans of the same place should produce similar descriptors, while different 
places should produce different ones.


2) How do we compare pairs and calculate performance?

Our baseline code handles this:
1. Give scan A to LoGG3D → descriptor A.
2. Give scan B to LoGG3D → descriptor B.
3. Use cosine_distance() to compare the descriptors. Smaller distance suggests the same place.
4. Compare that prediction with the correct label in your CSV, previously created using the robot’s positions.
5. Calculate AUC, F1, precision, and recall using our functions imported from scan_context_baseline.py.
So,the Logg3D-Net authors’ repo produces the descriptors. Our baseline compares them and evaluates the results.

"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

from scan_context_baseline import (
    get_site_folders,
    load_scan_points,
    find_labels_file,
    compute_roc_auc,
    compute_best_f1,
)


def parse_command_line_arguments():
    
    """
    This function reads the settings given on the command line when the
    script is started. There are five. The repo folder says where the
    LoGG3D-Net code was cloned, and the checkpoint says which published
    weights file to load, given relative to the repo folder unless it is a
    full path. The voxel size is the cube size the network uses to prepare
    each scan, and it is kept at 0.10 metres because our scans are already
    voxelized at that size and the authors used the same size. The intensity
    value is the constant that fills the fourth number of every point,
    since our scans have no intensity. The maximum pairs setting is for a
    quick first check: when it is above zero, only that many randomly chosen
    pairs are scored for each site, so a first run takes a minute instead of
    hours, and zero means score every pair.
    """
    
    parser = argparse.ArgumentParser(
        description="Run the pretrained LoGG3D-Net as a baseline on the Oxford pairs."
    )
    
    parser.add_argument(
        "--repo-folder",
        default="LoGG3D-Net",
        help="Folder with the cloned LoGG3D-Net code, relative to this script.",
    )
    
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mulran_10cm/2021-09-14_08-59-00_3n24h_MulRan_v10_q29_4s_263039.pth",
        help="Weights file, relative to the repo folder unless it is a full path.",
    )
    
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Voxel size in metres, also the name of our voxelized scan folders.",
    )
    
    parser.add_argument(
        "--intensity-value",
        type=float,
        default=0.0,
        help="Constant used for the missing intensity number of every point.",
    )
    
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Score only this many random pairs per site for a quick check, 0 means all.",
    )
    
    return parser.parse_args()





def load_model(repo_folder, checkpoint_path):
    
    """
    This function builds the LoGG3D-Net network from the authors' code,
    loads their published weights into it, moves it to the GPU and switches
    it to evaluation mode, which means it only produces descriptors and does
    no learning. It stops with a clear message if there is no GPU, if the
    code folder is missing or if the weights file is missing, because each
    of those would otherwise fail later with a confusing error. The authors'
    folder is added to Python's search path so their modules can be
    imported, and it returns both the model and the authors' function that
    converts a scan into the sparse format the network reads.
    """
    
    
    # if not torch.cuda.is_available():
    #     raise RuntimeError(
    #         "LoGG3D-Net needs an NVIDIA GPU with CUDA. Run this script on a GPU machine."
    #     )
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.isdir(repo_folder):
        raise FileNotFoundError(f"LoGG3D-Net code folder not found: {repo_folder}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")



    sys.path.insert(0, repo_folder)
    from models.pipeline_factory import get_pipeline
    from models.pipelines.pipeline_utils import make_sparse_tensor


    model = get_pipeline("LOGG3D")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, make_sparse_tensor






def compute_descriptor(scan_name, voxel_folder, model, make_sparse_tensor, voxel_size,
                       intensity_value, remembered_descriptors):
    """
    This function gets the descriptor of one scan, given only its name. It
    loads the voxelized scan, adds the constant intensity as a fourth number
    to every point because the network expects four, converts the scan into
    the sparse format with the authors' function, and runs it through the
    network with learning switched off, which gives the list of numbers that
    describes the place. Each scan appears in many pairs, so every
    descriptor is remembered and reused instead of being computed again. A
    scan with no points cannot be given to the network, so for it the
    function returns nothing, and the caller treats such a pair as
    maximally different.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if scan_name not in remembered_descriptors:
        points = load_scan_points(os.path.join(voxel_folder, f"{scan_name}.pcd"))

        if len(points) == 0:
            remembered_descriptors[scan_name] = None
        else:
            intensity_column = np.full((len(points), 1), intensity_value, dtype=np.float32)
            network_input = np.hstack([points.astype(np.float32), intensity_column])

            with torch.no_grad():
                sparse_input = make_sparse_tensor(network_input, voxel_size).to(device)
                descriptor, _ = model(sparse_input)

            remembered_descriptors[scan_name] = descriptor.cpu().numpy().reshape(-1)

    return remembered_descriptors[scan_name]





def cosine_distance(descriptor_a, descriptor_b):
    """
    This function measures how different two descriptors are, using the
    cosine distance, which is one minus the cosine of the angle between the
    two lists of numbers. It is near zero when the two descriptors point the
    same way, which means the scans look like the same place, and it grows
    as they point apart. This is the same distance the authors use in their
    own evaluation. If either descriptor is missing, or is all zeros so its
    direction is undefined, the function returns 1.0, the value for scans
    that do not match.
    """
    if descriptor_a is None or descriptor_b is None:
        return 1.0

    norm_product = np.linalg.norm(descriptor_a) * np.linalg.norm(descriptor_b)
    if norm_product == 0:
        return 1.0

    return float(1.0 - np.dot(descriptor_a, descriptor_b) / norm_product)






def score_pairs_for_site(site_folder, voxel_size, model, make_sparse_tensor, intensity_value,
                         max_pairs=0):
    """
    This function goes through every labeled pair of one site and gives it a
    difference score. It reads the site's label file, gets the descriptor of
    both scans of each pair, and records the cosine distance between them,
    so that a low number means the network thinks the two scans are the
    same place. If a maximum number of pairs is given and it is smaller than
    the number of pairs in the file, it first picks that many pairs at
    random with a fixed seed, so the quick check always uses the same pairs
    and contains both same-place and different-place examples, which the
    first rows of the file would not, because the file lists all the
    same-place pairs first. It returns the true labels and the scores in the
    same order, ready for the AUC and F1 measures, and it prints its
    progress every 250 pairs because a full run on a site can take a while.
    """
    
    
    voxel_folder = os.path.join(site_folder, f"individual_clouds_voxel_{voxel_size:.2f}m")
    pairs_table = pd.read_csv(find_labels_file(site_folder))

    if max_pairs > 0 and max_pairs < len(pairs_table):
        pairs_table = pairs_table.sample(n=max_pairs, random_state=0)

    remembered_descriptors = {}
    labels = []
    distances = []
    total_pairs = len(pairs_table)

    for pair_number, row in enumerate(pairs_table.itertuples(index=False), start=1):
        descriptor_1 = compute_descriptor(
            row.scan_1, voxel_folder, model, make_sparse_tensor, voxel_size,
            intensity_value, remembered_descriptors,
        )
        descriptor_2 = compute_descriptor(
            row.scan_2, voxel_folder, model, make_sparse_tensor, voxel_size,
            intensity_value, remembered_descriptors,
        )

        distances.append(cosine_distance(descriptor_1, descriptor_2))
        labels.append(int(row.label))

        if pair_number % 250 == 0 or pair_number == total_pairs:
            print(f"    scored {pair_number}/{total_pairs} pairs")

    return np.array(labels), np.array(distances)






if __name__ == "__main__":
    args = parse_command_line_arguments()

    project_root = os.path.dirname(os.path.abspath(__file__))
    repo_folder = args.repo_folder
    if not os.path.isabs(repo_folder):
        repo_folder = os.path.join(project_root, repo_folder)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(repo_folder, checkpoint_path)

    print(f"Loading LoGG3D-Net weights from {checkpoint_path}")
    model, make_sparse_tensor = load_model(repo_folder, checkpoint_path)

    site_folders = get_site_folders(project_root)
    results = {}

    for site_name, site_folder in site_folders.items():
        print(f"\nScoring pairs for {site_name}")
        start_time = time.time()
        try:
            labels, distances = score_pairs_for_site(
                site_folder, args.voxel_size, model, make_sparse_tensor, args.intensity_value,
                args.max_pairs,
            )
        except (FileNotFoundError, OSError) as error:
            print(f"  Skipping {site_name}: {error}")
            continue

        auc = compute_roc_auc(labels, distances)
        f1, precision, recall = compute_best_f1(labels, distances)
        results[site_name] = (auc, f1, precision, recall)
        print(f"  finished in {time.time() - start_time:.0f} s")





    print("\n \n")
    print("LoGG3D-Net baseline results (pretrained, no training on our data)")
    print(
        f"Settings: checkpoint {os.path.basename(checkpoint_path)}, voxel {args.voxel_size} m, "
        f"constant intensity {args.intensity_value}"
    )
    
    
    
    
    print("---" * 100)
    for site_name, (auc, f1, precision, recall) in results.items():
        print(
            f"{site_name:22s} AUC {auc:.3f} | best F1 {f1:.3f} "
            f"(precision {precision:.3f}, recall {recall:.3f})"
        )



# wytham_data            AUC 0.806 | best F1 0.612 (precision 0.554, recall 0.683)
# stein-am-rhein_data    AUC 0.764 | best F1 0.560 (precision 0.522, recall 0.604)
# forest-of-dean_data    AUC 0.923 | best F1 0.806 (precision 0.943, recall 0.704)
# evo_data               AUC 0.868 | best F1 0.666 (precision 0.609, recall 0.735)
# (venv) shahmeer@ShahmeersLaptop Baselines %