# PFDiff-Dehaze

Official PyTorch implementation of **PFDiff: Physics- and Frequency-Guided Residual Diffusion for Remote Sensing Image Dehazing**.

PFDiff-Dehaze restores paired hazy remote sensing images by combining a haze-aware restoration U-Net, physics/frequency priors, and conditional residual diffusion.

## Repository Structure

```text
.
├── configs/
│   └── pfdiff_full.json
├── datasets/
│   └── statehaze.py
├── models/
│   ├── backbone.py
│   ├── diffusion.py
│   └── pfdiff_dehaze.py
├── utils/
├── train.py
├── test.py
├── README.md
└── LICENSE
```

Only the core training and inference entry points are included:

- `train.py`: train PFDiff-Dehaze on paired hazy/clear data.
- `test.py`: run inference and compute PSNR/SSIM on a paired test split.

## Requirements

The code was developed for Python 3.10+ and PyTorch. Install the main dependencies with your preferred CUDA-compatible PyTorch build:

```bash
pip install torch pillow
```

Optional YAML configuration support requires:

```bash
pip install pyyaml
```

## Dataset Format

The dataset root must contain paired `hazy` and `clear` folders for each split. Files are matched by basename after removing the extension.

```text
DatasetRoot/
├── train/
│   ├── hazy/
│   └── clear/
└── test/
    ├── hazy/
    └── clear/
```

Example:

```text
train/hazy/0001.png
train/clear/0001.png
test/hazy/0001.png
test/clear/0001.png
```

## Datasets and Checkpoints

The released code supports paired remote sensing dehazing datasets organized with the folder structure above.

| Resource | Link | Access Code |
| --- | --- | --- |
| HazeRS45 dataset | [Baidu Netdisk](https://pan.baidu.com/s/1u8EKXjk1DTLT7pldPJuRmg) | `bi2n` |
| RSID dataset | [Trinity-Net repository](https://github.com/chi-kaichen/Trinity-Net) | N/A |
| Pre-trained checkpoints for RSID | [Baidu Netdisk](https://pan.baidu.com/s/13KagcbEW2BPv-6KXL9ahOQ?pwd=xb8r) | `xb8r` |

After downloading a dataset, arrange it as:

```text
data/
└── HazeRS45/ or RSID/
    ├── train/
    │   ├── hazy/
    │   └── clear/
    └── test/
        ├── hazy/
        └── clear/
```

For example, set `--data-root data/HazeRS45` or `--data-root data/RSID` when running training or testing.

## Configuration

The default configuration is `configs/pfdiff_full.json`. Important fields include:

- `data.root`: dataset root path.
- `data.image_size`: random crop size used during training.
- `training.epochs`: number of training epochs.
- `training.batch_size`: training batch size.
- `training.learning_rate`: AdamW learning rate.
- `experiment.output_dir`: directory for checkpoints, logs, and preview images.
- `experiment.result_dir`: default test output directory.
- `model.sampling_steps`: DDIM-style residual diffusion sampling steps.

The full model uses:

- physics priors: dark channel and transmission proxy.
- frequency prior: Laplacian high-frequency residual.
- adaptive residual gate.
- residual diffusion refinement.

The default training schedule follows the paper setting of 200 epochs with batch size 24. Reduce `training.batch_size` if your GPU memory is limited.

## Training

Run training from the repository root:

```bash
python train.py \
  --config configs/pfdiff_full.json \
  --data-root /path/to/DatasetRoot \
  --output-dir experiments/pfdiff_full \
  --variant full \
  --device cuda:0
```

Resume from the latest checkpoint:

```bash
python train.py \
  --config configs/pfdiff_full.json \
  --data-root /path/to/DatasetRoot \
  --output-dir experiments/pfdiff_full \
  --resume latest \
  --device cuda:0
```

Initialize from an existing checkpoint:

```bash
python train.py \
  --config configs/pfdiff_full.json \
  --data-root /path/to/DatasetRoot \
  --output-dir experiments/pfdiff_full \
  --init-checkpoint /path/to/checkpoint.pth \
  --device cuda:0
```

Available variants:

- `baseline`: haze-aware restoration backbone only.
- `diffusion`: residual diffusion without physics/frequency priors.
- `physics`: diffusion with physics priors.
- `frequency`: diffusion with frequency prior.
- `full`: diffusion with physics and frequency priors.

Training writes:

```text
experiments/pfdiff_full/
├── config.snapshot.json
├── train_log.csv
├── checkpoints/
│   ├── latest.pth
│   └── best.pth
└── samples/
```

## Testing

Run evaluation with a trained checkpoint:

```bash
python test.py \
  --config configs/pfdiff_full.json \
  --checkpoint /path/to/rsid_pretrained_checkpoint.pth \
  --data-root data/RSID \
  --output-dir results/rsid_pretrained \
  --device cuda:0
```

Testing writes:

```text
results/pfdiff_full/
├── images/
├── test_metrics.csv
└── summary.md
```

You can override inference-time diffusion settings:

```bash
python test.py \
  --config configs/pfdiff_full.json \
  --checkpoint experiments/pfdiff_full/checkpoints/best.pth \
  --data-root /path/to/DatasetRoot \
  --sampling-steps 10 \
  --residual-scale 1.0
```

## Citation

If this repository is useful for your research, please cite the paper after the final bibliographic information is available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
