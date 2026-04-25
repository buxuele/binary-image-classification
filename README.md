# Saint George Binary Image Classification

## Overview

A reproducible pipeline for binary image classification: does an image depict **Saint George** (positive / `1`) or not (negative / `0`)?

The task is non-trivial. Positive samples span wildly different visual styles -- oil paintings, pencil sketches, sculptures, ancient manuscripts, and medals -- while the negative set is packed with deceptive look-alikes: knights on horseback, medieval battle scenes, and dragon-free warriors in armor. The pipeline addresses this through aggressive data augmentation, dynamic class-weight balancing, and systematic error analysis.

**Final metrics on the held-out test set:**

| Metric | Value |
|---|---|
| Accuracy | 92.51 % |
| ROC AUC | 0.9676 |
| Precision (positive) | 93.15 % |
| Recall (positive) | 88.42 % |
| F1-Score (positive) | 90.72 % |

## Repository Structure

```text
.
├── data/                        # Training data (not tracked by git)
│   ├── georges/                 #   Positive samples
│   └── non_georges/             #   Negative samples
├── example_imgs/                # Sample images for quick inference tests
├── notebooks/                   # Jupyter notebooks (full experiment history)
├── dataset.py                   # PyTorch Dataset class and data transforms
├── train.py                     # Training script (ResNet18, transfer learning)
├── inference.py                 # Single-image / batch prediction
├── resnet18_best.pth            # Pre-trained model weights
├── report.md                    # Detailed experiment report and error analysis
├── requirements.txt             # Python dependencies
└── Dockerfile                   # CPU-only Docker image for inference
```

## Installation

### Option A -- Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/buxuele/binary-image-classification
   cd binary-image-classification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install PyTorch**

   If you have an NVIDIA GPU, install the build that matches your CUDA toolkit for full hardware acceleration (example below targets CUDA 12.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
   For CPU-only, simply omit the `--index-url` flag.

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Option B -- Docker (no local environment needed)

Build the image:
```bash
docker build -t george-classifier:latest .
```

Run inference on a single image:
```bash
# Windows (CMD / PowerShell)
docker run --rm -v "%CD%":/app george-classifier:latest python inference.py example_imgs/a1.jpg

# Linux / macOS
docker run --rm -v $(pwd):/app george-classifier:latest python inference.py example_imgs/a1.jpg
```

## Usage

### Inference

Run the pre-trained model against one or more images:
```bash
python inference.py example_imgs/a1.jpg
python inference.py example_imgs/a1.jpg example_imgs/a2.jpg
```
Output includes the predicted class and the confidence probability.

### Training

#### 1. Prepare the data

Download the dataset from one of the following mirrors:

- [Google Drive](https://drive.google.com/file/d/1b0QhYGsEuYpNqXOpszcfY_UxV-_jhCAf/view?usp=sharing)
- [Tencent Weiyun](https://share.weiyun.com/A5rD4aAy)

Place the image folders under `data/`:
- `data/georges/` -- positive samples
- `data/non_georges/` -- negative samples

#### 2. Launch training

```bash
python train.py
```

The script performs a 70 / 15 / 15 stratified split, applies data augmentation, and trains a ResNet18 model with transfer learning. The best checkpoint is saved automatically as `resnet18_best.pth`.

## Model Architecture

The final model is **ResNet18** with transfer learning.

- **Why ResNet18?** After experimenting with heavier architectures (EfficientNet-B0 through B3), ResNet18 delivered the best accuracy-to-cost ratio -- 92.51 % accuracy in just 8 epochs (under 10 minutes of training).
- **Head modification:** the original 1000-class FC layer is replaced with a single-output linear layer for binary classification.
- **Loss function:** `BCEWithLogitsLoss` with a dynamic `pos_weight` derived from the class ratio (3340 negative vs. 2360 positive) to penalize false negatives.
- **Optimizer:** AdamW (lr = 1e-4, weight_decay = 1e-4) with cosine-annealing LR schedule.

For the complete experiment history, architecture comparisons, and error analysis, see [report.md](report.md).