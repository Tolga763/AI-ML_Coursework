# Alzheimer's Disease Classification from Brain MRI

A deep learning pipeline that classifies Alzheimer's disease severity from structural brain MRI scans. Two CNN architectures are developed and compared: A custom-built CNN and a fine-tuned ResNet50V2 transfer learning model, with the transfer learning approach achieving **82% accuracy** and a **macro-average AUC of 0.933**.

---

## Overview

Alzheimer's disease accounts for approximately two-thirds of dementia cases globally, yet early-stage diagnosis remains difficult. Subtle structural brain changes overlap with normal ageing and are hard to detect from MRI imaging alone. This project explores whether convolutional neural networks can reliably automate the classification of disease severity from 2D brain MRI scans, evaluating the trade-offs between building a model from scratch versus leveraging pre-trained ImageNet models.

**Classes:** No Impairment · Mild Impairment · Moderate Impairment

**Dataset:** 5,076 MRI images — inherently imbalanced (50.4% No, 35.3% Mild, 14.3% Moderate)

---

## Methodology

### Preprocessing
- Images resized to 224×224 pixels and rescaled to [0,1]
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation, clip limit=2.0) applied to enhance visibility of subtle MRI features
- Stratified 64/16/20 train/validation/test split with preprocessing fitted strictly to training data to prevent data leakage

### Class Imbalance
Computed class weights using scikit-learn's `compute_class_weight`, assigning higher penalties to underrepresented classes:

| Class | Weight |
|---|---|
| No Impairment | 0.66 |
| Mild Impairment | 0.94 |
| Moderate Impairment | 2.33 |

### Model 1 — Custom CNN (from scratch)
- Initial Conv2D block (64 filters) followed by four deep convolutional blocks (128 → 256 → 512 → 728 filters) using `SeparableConv2D` for parameter efficiency
- Each block includes ReLU activation, batch normalisation, and max-pooling
- Progressive dropout regularisation (rate 0.2–0.5) to reduce overfitting
- Data augmentation: horizontal flipping, random rotations, zoom, contrast adjustment, and translation (5% range)
- Trained for up to 50 epochs, learning rate 1e-3, batch size 32

### Model 2 — ResNet50V2 (Transfer Learning)
- Pre-trained on ImageNet; images rescaled to [−1, 1] to match ResNet's expected input
- Custom classification head: GlobalAveragePooling → Dense(256) → Dense(128) → Softmax(3), with dropout (rate=0.3)
- Light augmentation only (5% zoom/contrast) to preserve MRI spatial features
- **Two-phase training:**
  - *Warm-up (10 epochs, LR=1e-4):* base frozen, only classification head trained
  - *Fine-tuning (20 epochs, LR=1e-5):* top 20 ResNet layers unfrozen for domain adaptation

---

## Results

### Overall Performance

| Model | Accuracy | Weighted F1 | Macro F1 | AUC |
|---|---|---|---|---|
| Model 1 — Custom CNN | 61% | 0.61 | 0.58 | 0.846 |
| Model 2 — ResNet50V2 | **82%** | **0.82** | **0.81** | **0.933** |

### Per-Class F1 Scores

| Class | Custom CNN | ResNet50V2 |
|---|---|---|
| No Impairment | 0.71 | 0.85 |
| Mild Impairment | 0.49 | 0.79 |
| Moderate Impairment | 0.54 | 0.78 |

ResNet50V2 yielded a much more balanced performance across all three classes. The gap is most pronounced on minority classes (Mild and Moderate), where the custom CNN struggled most.

---

## Key Findings

**Training instability in the custom CNN.** Model 1 showed pronounced spikes in validation loss (particularly around epoch 10) and erratic accuracy curves throughout training. This behaviour is consistent with gradient explosion — a phenomenon where gradients become excessively large during backpropagation, causing unstable weight updates. Similar instability was reported by Zhang et al. (2024), who found that ReLU activation contributed to gradient explosions in their AD classification model (ADNet), and resolved it by switching to ELU (Exponential Linear Unit) activation.

**Transfer learning resolved minority class failures.** The custom CNN struggled most with Mild Impairment (F1: 0.49), frequently misclassifying these cases as either No Impairment (111 instances) or Moderate Impairment (65 instances). ResNet50V2 corrected this, correctly identifying 286 Mild and 123 Moderate Impairment cases. Pre-trained ImageNet features generalised effectively to MRI, mapping biological markers of brain atrophy that the custom CNN could not learn reliably from scratch.

**Mild overfitting in ResNet50V2.** Despite stronger performance, Model 2 showed signs of overfitting. Training accuracy reached ~1.0 while validation accuracy plateaued around 80%. This is likely attributable to the relatively small dataset (5,076 images) relative to the model's high parameter count. The temporary disturbance at epoch 10 (start of fine-tuning) was expected and resolved as training continued.

**CLAHE improved feature separability.** Applying adaptive histogram equalisation as a preprocessing step enhanced the contrast of subtle cortical and hippocampal structures in the MRI images, likely contributing to improved model performance (particularly for the nuanced boundary between Mild and No Impairment cases).

---

## Limitations & Future Work

**2D slices discard volumetric context.** AD involves 3D structural changes (hippocampal and cortical atrophy) that a 2D slice cannot fully capture. Future work should use patient-stratified 3D MRI volumes with 3D-CNN architectures.

**Explainability.** Neither model provides insight into *which* brain regions drive predictions. Integrating **Grad-CAM** (Gradient-weighted Class Activation Mapping) would spatially verify that predictions are driven by clinically plausible regions (e.g. ventricular enlargement, cortical thinning) rather than image artefacts.

**Activation function.** Replacing ReLU with **ELU** may improve training stability in the custom CNN by producing smoother gradients and reducing the risk of gradient explosion (as demonstrated by Zhang et al. (2024).

**Dataset size and diversity.** A larger and more demographically diverse dataset would reduce overfitting risk and help address potential algorithmic bias across different patient populations.

---

## Ethical Considerations

Deep learning models in clinical settings raise important ethical questions. These models function as "black boxes". Their decision-making processes are opaque, making it difficult for clinicians to explain, audit, or challenge predictions. In Alzheimer's diagnosis, an incorrect classification could delay treatment or cause unnecessary distress.

Algorithmic bias is also a concern: if training data underrepresents certain demographics (by race, gender, or age), the model may perform inequitably across populations. Furthermore, MRI data is highly sensitive patient information, requiring strong governance and data protection measures (guided by frameworks such as HIPAA) to prevent misuse or breach.

While the results here are promising, these models are not yet suitable for independent clinical deployment. Any real-world application would need to be paired with explainability tools, diverse training data, and a clear regulatory and accountability framework.

---

## How to Run

This notebook is designed to run on **Google Colab** with a GPU runtime (T4 recommended).

1. Open [Google Colab](https://colab.research.google.com/) and upload `MASTER_MRI_PREDICTIONS.ipynb`
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Download the Alzheimer's MRI dataset locally
4. Upload `AD_Dataset.zip` to your Google Drive
5. Update `ZIP_PATH` in the notebook to your Drive path
6. Run all cells within the script 

### Dependencies

```bash
pip install tensorflow scikit-learn opencv-python matplotlib seaborn numpy pandas pillow tqdm
```

---

## Repository Structure

```
AI-ML_Coursework/
├── MASTER_MRI_PREDICTIONS.ipynb   # Full pipeline: EDA → preprocessing → training → evaluation
├── AI_ML coursework.pdf           # Full written report with methodology and discussion
└── README.md
```

---

## Academic Context

Developed as coursework for an AI/ML module — King's College London, 2024/25.
