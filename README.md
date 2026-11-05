# Calibrated Confidence for Kilometre-Scale Feed-Forward SLAM (mid submission code)

Kavish Vora, Pranav Srivastava, Mehrish Khan (IIIT Hyderabad), Mobile Robotics, Monsoon 2026.
Full repository with results: https://github.com/Pranav-S05/MR_Proj

VGGT-Long runs VGGT on overlapping windows of a long monocular drive, joins neighbouring windows with Sim(3) seams,
closes loops, and solves a pose graph in which every edge has the same weight. This code measures whether VGGT's
per-pixel confidence predicts the real error of pixels, seams and loops, turns it into calibrated 7x7 edge
information matrices (fitted on KITTI 05/07), and compares eight weighting schemes on the held-out KITTI 00/02.
No network is trained.

## Code layout

| Path | What it does |
|---|---|
| `mr_gpu/run_vggt.py` | GPU: VGGT on overlapping windows and loop windows (SALAD retrieval as in VGGT-Long), memory sweep, convention checks; writes `cache/kitti/<seq>/` |
| `mr_cpu/pipeline.py` | CPU: all steps after inference: `edges labels audit qual fit tune solve upstream report` |
| `mr_cpu/alignment.py` | VGGT-Long's window alignment (confidence-weighted IRLS Sim(3)) and per-edge features |
| `mr_cpu/edges.py`, `labels.py` | seam and loop edge table; ground-truth window placements and 7-DoF edge errors |
| `mr_cpu/depth_audit.py`, `kitti.py` | pixel audit of VGGT's depth and point heads against LiDAR; KITTI calibration, camera-2 ground truth, LiDAR projection |
| `mr_cpu/calibration.py`, `weighting.py` | isotonic maps, reliability, power law; weighting schemes (a), (a'), (b) to (g) |
| `mr_cpu/backend.py`, `lie.py` | sparse Levenberg-Marquardt Sim(3) pose graph with Huber / DCS kernels; batched Sim(3) maths |
| `mr_cpu/evaluation.py` | trajectory assembly, Sim(3)-aligned ATE, ATE vs length, KITTI t_rel / r_rel; command line tool to score any trajectory |
| `mr_cpu/report.py`, `qualitative.py` | summary tables, figures and qualitative figures |
| `mr_cpu/geometry.py`, `trajectory.py`, `cache.py` | shared geometry, pose file reading, cache format and VGGT-Long's window / loop layout |
| `mr_cpu/synthetic.py` | synthetic frontend with the same cache format, for testing without a GPU |
| `scripts/run_gpu_mid.sh` | the GPU part: memory sweep, smoke test, VGGT on 07 05 00 02, upstream VGGT-Long runs |
| `scripts/run_upstream.sh` | runs the original `vggt_long.py` at one window size and copies what is scored |
| `scripts/check_revisit_ground.py` | checks KITTI's ground truth at loop revisits with LiDAR (needs the IIIT network) |
| `configs/kitti_mid.json` | the experiment: fit on 05/07, test on 00/02 |
| `tests/test_pipeline.py` | 20 automated checks |

Weighting schemes: (a) VGGT-Long's equal weights; (a') equal weights in our residual convention (control);
(b) one fitted variance per block (rotation, translation, scale); (c) (b) scaled by squared mean confidence;
(d) isotonic map from the alignment residual; (e) isotonic map from mean point confidence (ours); (f) oracle using
the measured edge errors (reference only); (g) per-pixel power law of N. Deng et al. Each runs without loops, with
loops, and with Huber or DCS on loop edges.

## 1. Data

```bash
mkdir -p data && cd data
B=https://s3.eu-central-1.amazonaws.com/avg-kitti
wget $B/data_odometry_calib.zip $B/data_odometry_poses.zip
wget $B/data_odometry_color.zip        # 65 GB, all sequences
wget $B/data_odometry_velodyne.zip     # 80 GB, needed for the LiDAR audit
unzip -q data_odometry_calib.zip && unzip -q data_odometry_poses.zip
unzip -q data_odometry_color.zip    'dataset/sequences/0[0257]/image_2/*'
unzip -q data_odometry_velodyne.zip 'dataset/sequences/0[0257]/velodyne/*'
mv dataset kitti && cd ..
```
Result: `data/kitti/poses/SS.txt` and `data/kitti/sequences/SS/{image_2,velodyne,calib.txt,times.txt}`.

## 2. GPU part (Linux, CUDA 12 driver, 16 GB GPU)

```bash
git clone https://github.com/DengKaiCQ/VGGT-Long.git external/VGGT-Long
git -C external/VGGT-Long checkout c160869d1d99c96bb227f414afb3bc68c29c9a76
pixi install                           # environment from pixi.toml / pixi.lock
mkdir -p external/VGGT-Long/weights && cd external/VGGT-Long/weights
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt                          # 5 GB
wget https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
cd ../../..
pixi run gpu-mid
```
This runs the memory sweep (`results/memory_sweep.csv`), a smoke test that must print PASS for all four convention
checks, VGGT on sequences 07, 05, 00, 02 with windows of 32 frames and 16 frames overlap (about 2 s per window on an
RTX A4000, about 45 MB of cache per window), and the original VGGT-Long at window/overlap 30/15, 60/30, 32/16 and 75/30.

## 3. CPU part (any machine)

```bash
pixi run pipeline        # or: python -m venv .venv; pip install -r requirements-cpu.txt;
                         #     python -m mr_cpu.pipeline --config configs/kitti_mid.json
```
Outputs go to `results/mid/`: `summary.md` (all tables), `tables/*.csv`, `figures/*.png` and
`figures/qualitative/*.png`, `metrics.csv`, `upstream.csv`. Single steps: `--steps fit tune solve report`.
With little disk, run `--fit SS --test SS --steps edges labels audit qual` per sequence and then delete that
sequence's `cache/kitti/SS/windows` and `loops`; the later steps only read `manifest.json`.

Score any trajectory file with the same metrics:
`python -m mr_cpu.evaluation --estimate <camera_poses.txt> --seq 07 --evo`.

## 4. Tests and a check without a GPU

```bash
python -m unittest discover -s tests          # tests that need KITTI poses/calibration are skipped without data/
python -m mr_cpu.synthetic --seq 07 --out cache/synthetic/07   # likewise 05, 00, 02
python -m mr_cpu.pipeline --config configs/synthetic.json      # whole pipeline on a synthetic frontend
```
