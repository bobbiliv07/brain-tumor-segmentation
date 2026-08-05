# Brain Tumor Segmentation

A final year project focused on automated **brain tumor segmentation** from MRI scans using deep learning workflows in Jupyter Notebooks.

---

## Project Objective

The goal of this project is to:

- Segment brain tumor regions from multi-modal MRI images.
- Support reproducible preprocessing, model development, and evaluation in notebooks.
- Build a foundation for research and clinical-assistive AI workflows.

---

## Tech Stack

- **Language/Environment:** Python (via Jupyter Notebook)
- **Core Workflow:** Data preprocessing, model training, validation, and inference
- **Data Source:** BraTS dataset from Synapse

> Note: Specific libraries (e.g., TensorFlow/PyTorch, NumPy, OpenCV, MONAI) can be listed here once finalized.

---

## Dataset

This project uses the **BraTS (Brain Tumor Segmentation) dataset**.

### Download Instructions

1. Visit **Synapse**: https://www.synapse.org/
2. Create/login to your account.
3. Search for the latest BraTS challenge dataset.
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

> If your notebook expects a different directory layout, update paths accordingly in the notebook cells.

---

## Repository Structure

- `notebooks/` – Jupyter notebooks for preprocessing, training, and evaluation.
- `data/` – Local dataset storage (not recommended to commit raw medical images).
- `outputs/` – Trained models, generated predictions, visualizations, and logs.

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
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > If `requirements.txt` is not added yet, install dependencies used in your notebook manually.

4. **Prepare dataset**
   - Download BraTS from Synapse.
   - Place it under `data/brats/` (or update notebook paths).

5. **Launch Jupyter and run notebooks**
   ```bash
   jupyter notebook
   ```
   Open your project notebook and run cells in sequence.

---

## Results

Add your best outputs here for presentation quality:

- **Segmentation Metrics:** Dice Score, IoU, Precision, Recall
- **Qualitative Outputs:** Input MRI vs Ground Truth vs Predicted Mask
- **Model Performance Plots:** Loss and accuracy curves

### Sample Metrics (Placeholder for Demo)

| Metric | Value |
|---|---:|
| Dice (WT) | 82.4% |
| Dice (TC) | 79.8% |
| Dice (ET) | 76.9% |
| **Mean Dice** | **79.7%** |
| IoU (Mean) | 72.6% |

> These are placeholder values for documentation format only. Replace with actual experimental results.

---

## Future Improvements

- Add modular training scripts outside notebooks.
- Add experiment tracking (e.g., TensorBoard / Weights & Biases).
- Perform hyperparameter tuning and model ensembling.
- Containerize with Docker for reproducible deployment.
- Add API or lightweight UI for inference demos.

---

## Compliance & Usage Notes

- BraTS data access may require registration and approval via Synapse.
- Follow all dataset licensing, citation, and usage restrictions.
- Avoid publishing restricted raw medical imaging data in public repositories.

---

## Author

- **bobbili vamshi**

---

## Acknowledgements

- BraTS Challenge Organizers
- Synapse platform for dataset distribution
