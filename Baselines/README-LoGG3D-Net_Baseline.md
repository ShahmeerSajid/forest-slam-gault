# LoGG3D-Net Baseline: Setup and Run Instructions

## What this is

LoGG3D-Net is a published neural network that turns one LiDAR scan into a short list of numbers, called a descriptor. Two scans of the same place should get descriptors that are close to each other, and scans of different places should get descriptors that are far apart. We use it as a third baseline next to Scan Context and ICP. The file `logg3d_baseline.py` loads the network with the authors' released weights, turns every scan of our labeled Oxford pairs into a descriptor, measures the distance between the two descriptors of each pair, and reports ROC AUC and best F1 for each of the four forest sites. No training happens in this step. The weights come from the authors and were trained on city data (KITTI and MulRan), so how well they work on forests was an open question, and the answer is reported in the results section below.

The script has now been run with the real network on a Mac, on the CPU, and it produced results for all four sites. The steps that were needed to make TorchSparse work on a Mac are written in Step 4. A GPU machine is still the better choice for the full runs and is needed for fine-tuning.

## What you need

Best is a computer with an NVIDIA GPU and CUDA, such as William's machine. A Mac also works, but only on the CPU, which is much slower, because TorchSparse does not support the Mac GPU (MPS). The script picks the GPU if there is one and the CPU otherwise, and prints which one it chose. On a Mac, TorchSparse has to be built from source without CUDA and needs three small fixes, which are listed in Step 4. This route has been tested on an Apple Silicon Mac with Python 3.12. You also need the project folder with the four site folders (`wytham_data`, `stein-am-rhein_data`, `forest-of-dean_data`, `evo_data`), each holding its `slam_poses.csv`, its `..._loop_closure_pairs.csv` and its voxelized scan folder `individual_clouds_voxel_0.10m`.

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

On a Mac the plain install above fails, and the following steps are what worked. TorchSparse's setup file builds a CPU-only version when there is no CUDA, but three things in it break on a Mac. First, Apple's compiler does not accept the `-fopenmp` flag, which only turns on multi-core compiling, so the two flags that use it are swapped for harmless ones. Second, pip builds in an isolated space that cannot see PyTorch, so the install needs `--no-build-isolation`. Third, the CPU code asks PyTorch for `long` numbers, but on a Mac the 64-bit type is `long long`, and PyTorch does not provide the `long` version, so the import fails with a "symbol not found" error. Changing `long` to `int64_t` in two files fixes it.

```
brew install google-sparsehash libomp
git clone --branch v1.4.0 https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
sed -i '' 's/-fopenmp/-Wno-unused-variable/; s/-lgomp/-O3/' setup.py
sed -i '' 's/data_ptr<long>/data_ptr<int64_t>/g; s/long \*out/int64_t *out/; s/long int \*out/int64_t *out/' torchsparse/backend/hash/hash_cpu.cpp torchsparse/backend/others/query_cpu.cpp
export CPLUS_INCLUDE_PATH=/opt/homebrew/include
pip install --no-build-isolation .
```

The torchsparse folder is only needed for building, so keep it outside the project folder and out of git. Our script `logg3d_baseline.py` must also not stop when there is no NVIDIA GPU, and it must move the network and the scans to the chosen device with `.to(device)` and not with `.cuda()`. If an error mentions "Torch not compiled with CUDA enabled", a `.cuda()` call is still in the file, and `grep -n cuda logg3d_baseline.py` finds it. When it runs on the CPU, the output also shows many FutureWarning lines about old PyTorch function names, which can be ignored.

## Step 5: Run it

Always run a small test first. It scores only 40 random pairs per site, which takes a few minutes on a Mac CPU and shows whether the model loads and the chain works. The numbers from this test are only a rough guide, because 40 pairs holds about 10 positives, so one or two pairs change the score a lot. The test gave AUC between 0.86 and 0.94 on the four sites, which was too optimistic compared with the full run below.

```
python3 logg3d_baseline.py --max-pairs 40
```

If this works, run the full version, which scores every pair.

```
python3 logg3d_baseline.py
```

The script prints, for each of the four sites, the number of pairs, the ROC AUC, and the best F1 with its precision and recall, in the same layout as the Scan Context and ICP results so the three can go in one table. Useful options are `--repo-folder` if the repository is not called `LoGG3D-Net`, `--checkpoint` to choose another weight file, `--voxel-size` (default 0.10, which is what the network was trained with), and `--intensity-value`.

## Results

These are the results of the full run, with the pretrained MulRan weights, no training on our data, a voxel size of 0.1 m and a constant intensity of 0.0, scored on all labeled pairs of each site.

| Site | ROC AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|
| wytham_data | 0.806 | 0.612 | 0.554 | 0.683 |
| stein-am-rhein_data | 0.764 | 0.560 | 0.522 | 0.604 |
| forest-of-dean_data | 0.923 | 0.806 | 0.943 | 0.704 |
| evo_data | 0.868 | 0.666 | 0.609 | 0.735 |

The weights trained on city data separate same-place pairs from different-place pairs well on forests, with AUC between 0.76 and 0.92 and no training on our data. Stein-am-Rhein and Wytham are the hardest sites, and Forest of Dean is the easiest, but it also has the fewest pairs (284), so its score is the least stable. The AUC is the number to compare across methods, because it does not depend on the 3:1 ratio of negatives to positives that the best F1 depends on. These numbers still say little about the real project question for three reasons. The negatives are random scans more than 1.5 m apart, which are easy to tell apart from a revisit. All pairs come from one site and one drive, with heavy overlap between the two scans of a positive pair. And the test does not measure cross-site transfer, which is the point of the Oxford to Gault experiment.

## Things to know before trusting the numbers

The network expects four numbers per point, which are x, y, z and the LiDAR intensity. Our voxelized Oxford scans only have x, y and z, so the script fills the fourth number with a constant. The default is 0, while the authors' own training loader for custom data uses 1. When we compare the pretrained model with the fine-tuned model, both must use the same constant, so pick one value and use it for every run. The pretrained weights saw real intensity values on city data, so this is a difference from their training conditions and it should be written in the report. Whether the raw Oxford scans carry intensity, which could be kept through voxelization, has not been checked.

## Step 6 (later): Fine-tuning on Oxford

Fine-tuning means starting from these same weights and training a little more on our Oxford scans, then scoring the new weight file with the same script. Two helper scripts prepare it. `patch_logg3d_for_oxford.py` makes four small edits to the cloned repository, because the authors' training code cannot train on a custom dataset as released. `make_logg3d_training_pickle.py` builds the training file from our poses, and it leaves one site out so that site stays unseen for testing. Run them like this.

```
python3 patch_logg3d_for_oxford.py --repo-folder LoGG3D-Net
python3 make_logg3d_training_pickle.py --hold-out wytham_data
```

The patch script makes six edits, which also include making the `--train_pickles` setting readable from the command line and saving every epoch to its own checkpoint file, since the authors' code overwrites one file each epoch. Before any GPU time is used, check the whole chain on the CPU with a few real training steps. The script prints the size of each batch and the three loss values, which is a good way to see what one training step is.

```
python3 cpu_smoke_test_logg3d.py --pickle oxford_train_holdout_wytham_data.pickle --data-root .. --checkpoint LoGG3D-Net/checkpoints/mulran_10cm/2021-09-14_08-59-00_3n24h_MulRan_v10_q29_4s_263039.pth --steps 10
```

The real run goes through `finetune_logg3d.py`, which needs a name and a written reason for every run, saves the settings in `runs/<name>/config.json`, and copies the whole training output into `runs/<name>/train.log`. Anyone who runs it, on any machine, leaves the same record behind. A first run, with Wytham held out, looks like this, and `--dry-run` prints the command without training.

```
python3 finetune_logg3d.py --run-name wytham_lr1e-4 --hold-out wytham_data --learning-rate 1e-4 --epochs 5 --reason "small rate keeps the pretrained weights"
```

None of the training code has been run with the real network yet, only checked for syntax and tested on fake files, so the first smoke test may show errors that need fixing. Each saved epoch is scored with `python3 logg3d_baseline.py --checkpoint <epoch file>`, and only on the held-out site, since the other three sites were part of its training. The score of each epoch is compared with the pretrained score on the same site.

## Git

Add this to `.gitignore` before committing, so the repository and the weights are never uploaded.

```
LoGG3D-Net/
*.pth
*.pickle
```
