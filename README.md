# 🧠 CNN Concrete Classifier – USC ISE 529 Predictive Analytics (HW6)

This repository contains my implementation of a **Convolutional Neural Network (CNN)** model for classifying the **Concrete Dataset**, developed as part of **Homework 6** for *ISE 529: Predictive Analytics (Summer 2025)* at the **University of Southern California**.

All experiments were executed on the **USC CARC High Performance Computing (HPC)** cluster using a SLURM batch job.

---

## 📁 Repository Structure

```
cnn-concrete-classifier-hpc/
│
├── main.py              # Main CNN implementation (TensorFlow/Keras)
├── vraj.slurm                 # SLURM job submission script for HPC
├── vraj_job.out               # HPC execution output log
│
├── cnn_results.txt            # Final test accuracy and classification metrics
├── training_history.txt       # Recorded loss and accuracy for each epoch
│
├── confusion_matrix.png       # Confusion matrix visualization
├── Accuracy Curves.png        # Training/validation accuracy and loss curves
│
├── HW6.pdf                    # Homework instructions and dataset details
│
└── data/                      # Dataset folder (not included — see link below)
```

---

## 🧩 Project Overview

The task was to design and train a **Convolutional Neural Network (CNN)** using `TensorFlow` and `Keras` to classify images from the **Concrete Dataset**.  
The project emphasizes:
- Model design and training for multi-class classification  
- GPU-accelerated computation via HPC  
- Evaluation using confusion matrices and learning curves  
- Performance reporting (accuracy, precision, recall, F1-score)

Dataset source: [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

---

## ⚙️ Model Performance

The model achieved near-perfect classification results:

| Metric | Training | Validation | Testing |
|--------|-----------|-------------|----------|
| Accuracy | 99.6% | 99.8% | **100.0%** |

**Confusion Matrix:**
```
[[3994    6]
 [   5 3995]]
```

**Classification Report:**
```
Precision  Recall  F1-Score  Support
1.00       1.00    1.00      8000
```

---

## 📈 Training History (Excerpt)

| Epoch | Loss | Accuracy | Val_Loss | Val_Accuracy |
|-------|------|-----------|-----------|---------------|
| 1 | 0.1021 | 0.9664 | 0.0440 | 0.9863 |
| 5 | 0.0301 | 0.9911 | 0.0122 | 0.9965 |
| 10 | 0.0307 | 0.9916 | 0.0090 | 0.9975 |
| 20 | 0.0139 | 0.9962 | 0.0047 | 0.9986 |

---

## 🚀 Running on HPC

Run the model using the SLURM scheduler:

```bash
sbatch vraj.slurm
```

This script:
1. Loads the necessary TensorFlow modules and dependencies  
2. Activates the Python virtual environment  
3. Executes the `Vraj_Patel.py` training script on GPU nodes

---

## 🧠 Technologies Used

- Python 3.9  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- SLURM Workload Manager  
- USC CARC HPC Cluster

---

## 📚 Reference

Concrete dataset source: [Mendeley Data – 5y9wdsg2zt/2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

---

© 2025 Vraj Patel | USC ISE 529 Predictive Analytics
