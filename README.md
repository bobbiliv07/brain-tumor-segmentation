# Brain Tumor Segmentation

A final year project focused on automated **brain tumor segmentation** from multi-modal MRI scans using deep learning workflows in Jupyter Notebooks.

The project trains a 3D deep-learning model to identify tumor regions voxel by voxel. Instead of only predicting whether a tumor is present, the system produces a complete 3D segmentation mask showing the location and type of tumor tissue.

---

## Project Objective

The goal of this project is to:

- Segment brain tumor regions from multi-modal MRI images.
- Separate the tumor into clinically relevant subregions.
- Support reproducible preprocessing, model development, and evaluation.
- Reduce the amount of manual work required for tumor annotation.
- Build a foundation for research and clinical-assistive AI workflows.

This system is intended to assist medical experts and researchers. It is not intended to replace radiologists, oncologists, or clinical decision-making.

---

## Project Workflow

The complete project follows this pipeline:

```text
BraTS MRI data
      ↓
Dataset discovery and validation
      ↓
MRI preprocessing and label conversion
      ↓
Training and validation split
      ↓
SwinUNETR model training
      ↓
Sliding-window inference
      ↓
Predicted 3D tumor mask
      ↓
Dice-score evaluation and visualization
```

The project also prepares BraTS data for an nnU-Net pipeline. The evaluation notebook compares predictions from **SwinUNETR** and **nnU-Net**.

---

## Tech Stack

- **Language/Environment:** Python and Jupyter Notebook
- **Deep Learning:** PyTorch
- **Medical AI Framework:** MONAI
- **Main Model:** SwinUNETR
- **Comparison Model:** nnU-Net
- **Medical Image Format:** NIfTI (`.nii` and `.nii.gz`)
- **Data Processing:** NumPy, pandas, nibabel, SciPy, OpenCV
- **Evaluation and Visualization:** scikit-learn, matplotlib, seaborn
- **Hardware:** CUDA-enabled NVIDIA GPU recommended

The main training notebook used an NVIDIA RTX A6000 GPU with approximately 51.5 GB of detected GPU memory.

---

## Dataset

This project uses the **BraTS brain tumor segmentation dataset** from Synapse.

Each patient case contains four aligned MRI modalities and one expert-provided segmentation label:

1. **T1** - structural brain information
2. **T1ce** - contrast-enhanced T1 MRI
3. **T2** - fluid-sensitive MRI information
4. **FLAIR** - useful for identifying edema and tumor-related abnormalities
5. **Segmentation mask** - the ground-truth tumor annotation

The project notebook discovered **1,350 valid patient cases**. In the SwinUNETR training workflow, the cases were divided into:

- **1,080 training cases**
- **270 validation cases**

The split used `random_state=42` for reproducibility.

### Download Instructions

1. Visit **Synapse**: https://www.synapse.org/
2. Create or log in to your account.
3. Search for the required BraTS challenge dataset.
4. Accept the data usage terms.
5. Download the required files.

### Recommended Local Folder Structure

```text
brain-tumor-segmentation/
├── data/
│   └── brats/
│       ├── train/
│       ├── val/
│       └── test/
├── notebooks/
│   └── brain_tumor_segmentation.ipynb
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── figures/
└── README.md
```

> The notebooks currently use local Windows paths. Update those paths before running the project on another computer.

---

## Segmentation Classes

The segmentation labels are converted into four classes:

| Label | Region | Description |
|---:|---|---|
| 0 | Background | Normal area outside the tumor |
| 1 | NCR/NCR-Net | Necrotic and non-enhancing tumor region |
| 2 | ED | Peritumoral edema |
| 3 | ET | Enhancing tumor region |

The original BraTS labels may contain the value `4`. The notebooks remap label `4` to label `3` so that the model uses class IDs from `0` to `3`.

---

## Main Model: SwinUNETR

The primary model is **SwinUNETR**, a 3D segmentation network provided by MONAI.

SwinUNETR combines:

- A **Swin Transformer encoder**, which learns relationships between different regions of the MRI volume.
- A **U-Net-style decoder**, which reconstructs a detailed segmentation mask.

The model receives four MRI modalities as input and predicts four classes for every voxel.

```text
Four MRI modalities
        ↓
Swin Transformer encoder
        ↓
Deep 3D feature representation
        ↓
U-Net decoder
        ↓
Four-class 3D segmentation mask
```

### Model Configuration

- **Input channels:** 4
- **Output classes:** 4
- **Training patch size:** `128 × 128 × 128`
- **Feature size:** 48
- **Trainable parameters:** approximately 62.2 million
- **Model checkpoint:** `outputs/best_model.pth`
- **Final weights:** `outputs/swinunetr_brats_final.pth`

---

## How the ML Model Ran

### 1. Data loading

The notebook scans each patient folder and searches for the four MRI modalities and the segmentation mask. The four MRI volumes are combined into one input tensor:

```text
(batch, 4 modalities, depth, height, width)
```

The ground-truth label is loaded as:

```text
(batch, 1, depth, height, width)
```

### 2. Preprocessing

The following preprocessing operations are applied:

- MRI volumes are loaded using MONAI.
- Channels are moved to the first dimension.
- BraTS labels are converted to the expected class IDs.
- Training patches of size `128 × 128 × 128` are extracted.
- Positive and negative regions are sampled so the model sees both tumor and background.
- MRI intensities are normalized using non-zero, channel-wise normalization.

### 3. Training

During training, the model receives MRI patches and predicts a class score for every voxel. The prediction is compared with the expert segmentation using `DiceCELoss`.

For each batch:

1. MRI patches are moved to the GPU.
2. SwinUNETR generates a segmentation prediction.
3. Dice plus cross-entropy loss is calculated.
4. Gradients are computed using backpropagation.
5. The AdamW optimizer updates the model parameters.
6. The training loss is recorded.

### 4. Loss function

The project uses `DiceCELoss`, which combines:

- **Dice loss:** encourages overlap between the predicted tumor and the ground-truth tumor.
- **Cross-entropy loss:** encourages the correct class prediction for each voxel.

This combination is useful because tumor tissue occupies a much smaller area than the background.

### 5. Optimization

The optimizer is **AdamW** with an initial learning rate of `1e-4` and weight decay of `1e-5`.

A cosine-annealing learning-rate scheduler gradually reduces the learning rate during the 20 training epochs. Mixed-precision training with `autocast` and `GradScaler` is also used to reduce GPU memory usage and improve performance.

### 6. Validation

Validation is performed every two epochs. Because full MRI volumes are larger than the training patch size, the notebook uses **sliding-window inference**:

1. The full volume is divided into overlapping `128 × 128 × 128` windows.
2. The model predicts each window.
3. The overlapping predictions are combined.
4. A complete 3D segmentation is produced.

The overlap is set to `0.5` to reduce artifacts at window boundaries.

### 7. Checkpoint saving

After validation, the mean Dice score is calculated. If the score improves, the model is saved as the best checkpoint:

```text
outputs/best_model.pth
```

The final model after all epochs is saved as:

```text
outputs/swinunetr_brats_final.pth
```

---

## Training Results

The SwinUNETR training loss decreased during the 20 epochs:

```text
Epoch 1:  0.8846
Epoch 5:  0.3994
Epoch 10: 0.3488
Epoch 15: 0.3192
Epoch 20: 0.3096
```

The best validation mean Dice score recorded in the training notebook was:

```text
Mean validation Dice: 0.7915
```

The final validation scores were:

```text
NCR: 0.6425
ED:  0.8778
ET:  0.8542
Mean: 0.7915
```

The lower NCR score indicates that the necrotic/non-enhancing region was more difficult for the model to segment than edema and enhancing tumor.

---

## Inference Process

After training, `test.ipynb` loads the saved SwinUNETR weights and performs inference on the MRI volumes.

For each case:

1. The four MRI modalities are loaded.
2. The modalities are stacked into one four-channel input.
3. Each modality is normalized.
4. Sliding-window inference is applied.
5. The model produces four output channels.
6. `argmax` selects the class with the highest score for each voxel.
7. The prediction is saved as a NIfTI segmentation file.

Each prediction contains values from `0` to `3` and represents a complete 3D tumor mask.

---

## Evaluation Metrics

The project evaluates three clinically relevant tumor regions.

### Enhancing Tumor, ET

```text
ET = label 3
```

### Tumor Core, TC

```text
TC = label 1 OR label 3
```

### Whole Tumor, WT

```text
WT = any label greater than 0
```

### Dice Score

The Dice score measures the overlap between the predicted segmentation and the ground-truth segmentation:

```text
Dice = 2 × intersection(prediction, ground truth)
       -------------------------------------------
       prediction volume + ground-truth volume
```

A score of `0` means no overlap, while `1` means perfect overlap.

The testing notebook also provides:

- Confusion matrices
- Dice-score distributions
- Error maps
- Ground-truth versus prediction visualizations
- Model comparison plots

---

## Model Comparison

The repository evaluates both nnU-Net and SwinUNETR predictions.

The evaluation notebook reported the following results:

| Model | ET Dice | TC Dice | WT Dice |
|---|---:|---:|---:|
| nnU-Net | 0.8466 ± 0.1778 | 0.8632 ± 0.1695 | 0.9177 ± 0.0680 |
| SwinUNETR test evaluation | 0.0303 ± 0.0466 | 0.0322 ± 0.0476 | 0.0612 ± 0.0569 |

The SwinUNETR training notebook recorded a much stronger validation result of `0.7915` mean Dice. The difference between the validation result and the separate test evaluation should be investigated before presenting the test score as the final SwinUNETR performance.

Possible causes include differences in:

- Preprocessing and normalization
- MRI modality ordering
- Label conventions
- Checkpoint loading
- Spatial orientation or affine information
- Training and inference file naming
- Training versus testing transformations

The file and shape checks in `test.ipynb` found 1,350 predictions, 1,350 ground-truth files, no missing ground truth files, and no shape mismatches. Therefore, the issue is more likely related to preprocessing, label semantics, or checkpoint compatibility than file matching.

---

## Importance of the Model

This model is important because it can:

1. **Reduce manual annotation time** by generating an initial segmentation for review.
2. **Provide detailed 3D tumor information** rather than only predicting tumor presence.
3. **Estimate tumor regions and volumes**, including edema, tumor core, and enhancing tumor.
4. **Support treatment planning research** and treatment-response monitoring.
5. **Improve reproducibility** by applying the same segmentation process to multiple scans.
6. **Support clinical-assistive AI workflows** when used with expert supervision.
7. **Demonstrate modern medical AI techniques**, including 3D transformers, U-Net decoding, GPU acceleration, mixed precision, and sliding-window inference.

The model should be considered a research and clinical-assistance system. It requires further validation on independent data before it can be used in a real clinical environment.

---

## Repository Structure

- `optimized_swinunetr_brats_fixed.ipynb` - Main SwinUNETR training and validation pipeline.
- `updated_brats_nnunet_notebook.ipynb` - BraTS preparation and nnU-Net preprocessing/training pipeline.
- `test.ipynb` - Inference, Dice evaluation, confusion matrices, visualizations, and model comparison.
- `monitor_training.py` - Displays updates from nnU-Net training log files.
- `requirements.txt` - Python dependencies.
- `data/` - Local dataset storage. Raw medical images should not normally be committed.
- `outputs/` - Model checkpoints, predictions, visualizations, and logs.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/bobbiliv07/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. **Create and activate a Python environment**

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The notebooks also require compatible installations of PyTorch, MONAI, and, for GPU training, a CUDA-enabled PyTorch build. These packages should be added to `requirements.txt` according to the target CUDA version.

4. **Prepare the dataset**

   - Download the BraTS dataset from Synapse.
   - Update `DATA_ROOT` in the notebooks.
   - Confirm that every case contains the four MRI modalities and a segmentation file.

5. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

6. **Run the notebooks**

   Run the cells in `optimized_swinunetr_brats_fixed.ipynb` in order to train SwinUNETR. Then use `test.ipynb` for inference and evaluation.

> The notebooks contain machine-specific Windows paths such as `C:\Users\Admin\Desktop\...`. These paths must be changed before running on another computer.

---

## Results

Add final presentation outputs here:

- Segmentation metrics for ET, TC, and WT
- Input MRI versus ground truth versus prediction
- Error maps
- Confusion matrices
- Training-loss curves
- Validation-Dice curves
- nnU-Net versus SwinUNETR comparison plots

Do not present the placeholder values below as final experimental results:

| Metric | Value |
|---|---:|
| Dice (WT) | 82.4% |
| Dice (TC) | 79.8% |
| Dice (ET) | 76.9% |
| **Mean Dice** | **79.7%** |
| IoU (Mean) | 72.6% |

> Actual results depend on the dataset split, preprocessing, model checkpoint, hyperparameters, and evaluation pipeline.

---

## Future Improvements

- Resolve the difference between SwinUNETR validation and test performance.
- Use exactly the same preprocessing pipeline for training, validation, and inference.
- Add an independent test split that is not used during training or validation.
- Add modular training and inference scripts outside notebooks.
- Add experiment tracking using TensorBoard or Weights & Biases.
- Perform hyperparameter tuning and model ensembling.
- Improve checkpoint and configuration management.
- Containerize the project with Docker.
- Add an API or lightweight user interface for inference demonstrations.
- Validate the model on external hospital data.

---

## Compliance & Usage Notes

- BraTS data access may require registration and approval through Synapse.
- Follow all dataset licensing, citation, and usage restrictions.
- Do not publish restricted raw medical imaging data in a public repository.
- Remove patient-identifying information from all data and outputs.
- This project is for research and educational purposes and is not a certified medical device.

---

## Author

- **bobbili vamshi**

---

## Acknowledgements

- BraTS Challenge Organizers
- Synapse platform for dataset distribution
- MONAI and PyTorch communities
- nnU-Net project contributors
