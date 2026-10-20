# LoGG3D-Net Baseline: Setup and Run Instructions

## What this is

LoGG3D-Net is a published neural network that turns one LiDAR scan into a short list of numbers, called a descriptor. Two scans of the same place should get descriptors that are close to each other, and scans of different places should get descriptors that are far apart. We use it as a third baseline next to Scan Context and ICP. The file `logg3d_baseline.py` loads the network with the authors' released weights, turns every scan of our labeled Oxford pairs into a descriptor, measures the distance between the two descriptors of each pair, and reports ROC AUC and best F1 for each of the four forest sites. No training happens in this step. The weights come from the authors and were trained on city data (KITTI and MulRan), so a low score on forests is an expected and reportable result, not a bug.


## What you need

A computer with an NVIDIA GPU and CUDA. The network cannot run on a Mac or on a CPU, and the script stops with a clear message if no GPU is found. This is the machine William uses, or any other GPU machine we get access to. You also need the project folder with the four site folders (`wytham_data`, `stein-am-rhein_data`, `forest-of-dean_data`, `evo_data`), each holding its `slam_poses.csv`, its `..._loop_closure_pairs.csv` and its voxelized scan folder `individual_clouds_voxel_0.10m`.

## Step 1: Put the scripts in the Baselines folder

The folder layout that works is the project folder with a `Baselines` folder inside it. In `Baselines` we keep `scan_context_baseline.py`, `icp_baseline.py` and `logg3d_baseline.py`. The scripts look for the site folders next to themselves first and then one folder up, so they work from `Baselines` without any change.

## Step 2: Clone the LoGG3D-Net code

The script needs the authors' code, because the network's definition and its data preparation function live in their repository. Clone it inside `Baselines`.

```
cd Baselines
git clone https://github.com/csiro-robotics/LoGG3D-Net.git
```

This creates `Baselines/LoGG3D-Net`. Do not push this folder to our GitHub, because it is someone else's code and it will hold the large weight files.

## Step 3: Download the weights

The authors' weights are one zip file of about 741 MB with seven checkpoints, six trained on KITTI and one on MulRan. Download it into the cloned folder and unzip it so that a `checkpoints` folder appears inside `LoGG3D-Net`. The link is quoted because it contains an `&` character.

```
cd LoGG3D-Net
curl -L -o checkpoints.zip "https://www.dropbox.com/scl/fi/ls0uyns1gv2q37frmy2p0/checkpoints.zip?rlkey=jt89tsxvgkghcww5v5xm3086x&dl=1"
unzip checkpoints.zip
ls -lh checkpoints/*
```

Each checkpoint file should be around 100 MB. If a file is only a few kilobytes, the download saved a web page instead of the weights, and the zip should be downloaded again. If the Dropbox link stops working, the authors' README lists a second link on cloudstor. Our own run uses the MulRan checkpoint by default, the path `checkpoints/mulran_10cm/2021-09-14_08-59-00_3n24h_MulRan_v10_q29_4s_263039.pth`. If the folder names after unzipping are different, pass the right path with `--checkpoint`.

## Step 4: Install the Python libraries

Use a fresh virtual environment on the GPU machine. Install PyTorch first, with the CUDA version that matches the machine, then the rest.

```
pip install torch torchvision
pip install numpy pandas scikit-learn open3d torchpack
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

TorchSparse 1.4.0 is the hardest part of the setup, because it is compiled against a specific PyTorch and CUDA version and often needs an older PyTorch than the newest one. The authors' README lists the versions they used, and the safest way is to match those. If the TorchSparse install fails, send the full error message. It also needs the `libsparsehash` system package on Linux.

## Step 5: Run it

Always run a small test first. It scores only 40 pairs per site, which takes a minute or two and shows whether the model loads and the numbers look sensible.

```
python3 logg3d_baseline.py --max-pairs 40
```

If this works, run the full version, which scores every pair.

```
python3 logg3d_baseline.py
```

The script prints, for each of the four sites, the number of pairs, the ROC AUC, and the best F1 with its precision and recall, in the same layout as the Scan Context and ICP results so the three can go in one table. Useful options are `--repo-folder` if the repository is not called `LoGG3D-Net`, `--checkpoint` to choose another weight file, `--voxel-size` (default 0.10, which is what the network was trained with), and `--intensity-value`.

## Things to know before trusting the numbers

The network expects four numbers per point, which are x, y, z and the LiDAR intensity. Our voxelized Oxford scans only have x, y and z, so the script fills the fourth number with a constant. The default is 0, while the authors' own training loader for custom data uses 1. When we compare the pretrained model with the fine-tuned model, both must use the same constant, so pick one value and use it for every run. The pretrained weights saw real intensity values on city data, so this is one more reason a low forest score is expected, and it should be written in the report.
