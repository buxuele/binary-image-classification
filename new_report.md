# Technical Report: Experiment History and Error Analysis

## 1. Executive Summary

This project successfully developed a highly accurate binary classifier to detect images of Saint George. After extensive experimentation across 6 iterations, the final ResNet18 model achieved the following metrics on the unseen test set:

* **Accuracy:** 92.51%
* **ROC AUC Score:** 0.9676
* **Precision (Positive Class):** 93.15%
* **Recall (Positive Class):** 88.42%
* **F1-Score (Positive Class):** 90.72%

An AUC of 0.9676 indicates that the model has excellent sorting and distinguishing capabilities, even when faced with highly deceptive medieval artwork. 

**Confusion Matrix (Test Set of 855 images):**

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 313 (TP) | 41 (FN) |
| **Actual Negative** | 23 (FP) | 478 (TN) |


## 2. Data Strategy and Challenges

Initial Exploratory Data Analysis (EDA) revealed severe challenges in the dataset geometry and semantics:

1. **Intra-class Variance:** The `georges` folder contains everything from grand 4000x3000 oil paintings to minimalist sketches, and even tiny 64x64 coin medals. 
2. **Deceptive Negatives:** The `non_georges` folder is heavily populated with horses, knights, and spears.
3. **Class Imbalance:** 3340 negative vs. 2360 positive samples (ratio ~ 1.4 : 1).

**Solutions applied:**
* **No data deletion:** Instead of randomly dropping negative samples to achieve balance, I utilized `pos_weight` in the `BCEWithLogitsLoss`. This allows the model to see all negative variations while heavily penalizing misses on the smaller positive class.
* **Stratified Split:** Data was strictly split into 70% Train, 15% Validation, and 15% Test sets to ensure consistent distributions across all splits.
* **Heavy Augmentation:** Random rotations, resized crops, and color jitters were applied to force the model to learn semantic features (e.g., the dragon) rather than memorizing rigid image compositions.


## 3. Experiment History and Architecture Choices

> All experiments use transfer learning rather than training from scratch. With only ~5,700 images — many of which are visually ambiguous even to a human — a randomly initialized CNN would severely overfit. Pre-trained ImageNet weights already encode low-level features (edges, textures, color distributions) that transfer directly to classical artwork. This decision trades "novelty" for reliability.

The reproducible experiments are mapped to the following notebooks:

| Notebook | Purpose |
|---|---|
| `a1-ResNet18-start.ipynb` | Baseline ResNet18 experiment |
| `a2-ResNet18-epochs-8.ipynb` | Augmentation + CosineAnnealingLR improvements |
| `a3-get-all-wrong-images.ipynb` | Error analysis and bad-case extraction |
| `a4-EfficientNet-B0.ipynb` | EfficientNet-B0 single-stage fine-tuning |
| `a5-EfficientNet-2-stages-batchsize-32.ipynb` | EfficientNet two-stage fine-tuning |
| `a6-ResNet18-FocalLoss-and-LabelSmoothing.ipynb` | Focal Loss and Label Smoothing experiments |

### Phase 1: Baseline Establishment (ResNet18)
* **Approach:** Transfer learning using a pre-trained ResNet18. Only the final layer was modified.
* **Result:** Reached ~91.4% validation accuracy rapidly.
* **Conclusion:** The pre-trained features (edge and texture detection) are highly applicable to classical artworks.

### Phase 2: Pursuit of Higher Accuracy (EfficientNet-B0 to B3)
* **Hypothesis:** A more modern architecture (EfficientNet) would capture finer details (like a dragon in a crowded battle scene).
* **Process:** Attempted both single-stage and two-stage fine-tuning (freeze then unfreeze).
* **Observation:** EfficientNet-B3 suffered severe performance degradation and caused GPU Out-Of-Memory (OOM) issues due to large batch sizes. 
* **Root Cause Analysis:** EfficientNet-B3 expects a larger input resolution (e.g., `300x300`+). Forcing it to process `224x224` images destroyed its pre-trained spatial assumptions. Furthermore, the large parameter count led to massive VRAM spikes during the unfreezing phase.

### Phase 3: Advanced Loss Functions (Focal Loss and Label Smoothing)
* **Hypothesis:** Using Focal Loss would force the model to focus on "hard" deceptive samples; Label Smoothing would counteract label noise.
* **Result:** Accuracy plateaued at 92.28%. 
* **Conclusion:** The bottleneck was not the loss function, but the physical limitation of the `224x224` resolution. Shrinking a complex ancient manuscript to 224 pixels turns the core subject (Saint George) into an unrecognizable blur.

### Experiment Result Comparison Table

| # | Architecture | Input | Epochs | Val Acc | Test Acc | AUC | Training Time |
|---|---|---|---|---|---|---|---|
| 1 | ResNet18 (baseline) | 224x224 | 5 | 91.40% | -- | -- | ~5 min |
| 2 | ResNet18 (augmentation + Scheduler) | 224x224 | 8 | 93.00% | 92.51% | 0.9676 | ~10 min |
| 3 | EfficientNet-B0 (single-stage) | 224x224 | 10 | ~90.0% | -- | -- | ~20 min |
| 4 | EfficientNet-B3 (two-stage) | 224x224 | 5+5 | degraded | -- | -- | OOM |
| 5 | ResNet18 + Focal Loss | 224x224 | 10 | 92.28% | -- | -- | ~12 min |
| 6 | ResNet18 + Label Smoothing | 224x224 | 10 | ~92.0% | -- | -- | ~12 min |


## 4. Time Allocation

Explanation of time spent during the project pipeline:

| Stage | Wall-clock Time | Outcome |
|---|---|---|
| EDA + data inspection | ~1 hour | Identified class imbalance, scale variance, and label noise |
| ResNet18 baseline (Notebook a1) | ~5 min training | 91.4% val accuracy proof-of-concept |
| ResNet18 improved (Notebook a2) | ~12 min training | 92.51% test accuracy (The Final chosen model) |
| Error analysis (Notebook a3) | ~30 min manual review | Isolated 3 specific hardware/software failure categories |
| EfficientNet experiments (a4, a5) | ~2 hours | Debugged OOM errors; No improvement over ResNet18 |
| Focal Loss & Label Smoothing (a6) | ~30 min | Marginal gain, determined not worth the added complexity |
| Code refactor + Docker + README | ~2 hours | Production-ready English pipeline components built |


## 5. Error Analysis (Bad Case Review)

I built an extraction tool to isolate misclassified images from the test set for manual review. Out of 855 test images, 64 were misclassified (41 False Negatives, 23 False Positives). 

Manual inspection revealed distinct "Natural Traps":
1. **The "Dominant Background" Trap:** Ancient manuscripts where Saint George is merely a tiny illustration in the margin. By squashing the image to 224x224, the model only sees background text/borders.
2. **The "Element Deception" Trap:** Images containing [Knight in Armor + Horse + Spear]. The model occasionally relies on this structural shortcut rather than actively searching for the [Dragon].
3. **Dataset Noise / Label Noise:** Several images in the negative dataset actually contain elements visually identical to Saint George (e.g., mislabeled medieval figures), confusing the BCE loss.


## 6. Cost-Benefit Analysis

In machine learning engineering, "Good enough is good enough." 

While spending hours training a heavier model at a `600x600` resolution might yield a 1-2% absolute accuracy improvement, the ResNet18 baseline achieved an impressive **92.51% accuracy and 0.9676 AUC in under 15 minutes of training**. 

Therefore, ResNet18 was selected as the final architecture. It provides an optimal balance: it is lightweight, lightning-fast in inference, handles `224x224` resolution perfectly, and delivers industrial-grade classification metrics with minimal hardware constraints.


## 7. Future Improvements

If hardware overhead and time constraints were removed, the following steps would likely push the classification accuracy beyond 95%:

1. **Higher Resolution Inputs:** Using ResNet50 or EfficientNet with `384x384` or `512x512` inputs to resolve the "Dominant Background" failures on intricate manuscripts.
2. **Model Ensembling:** Averaging the predictions of a CNN (excellent parameter-efficient spatial textures) and a Vision Transformer (ViT) (excellent global context understanding). In competition settings, this routinely adds 1-3% accuracy.
3. **Attention-based Cropping:** Use GradCAM to dynamically locate the model's region of attention, then auto-crop and re-classify at higher resolution. This directly addresses the "tiny subject in a large margin" failure mode.
