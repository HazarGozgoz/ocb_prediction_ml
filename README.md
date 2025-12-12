# OCB Prediction ML: A Machine Learning Pipeline for CSF Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Published in AJCP](https://img.shields.io/badge/Published%20in-AJCP-red)](https://academic.oup.com/ajcp)

## 📌 Overview
This repository contains the official implementation of the machine learning pipeline described in the paper: **"A Machine Learning Model for Predicting Oligoclonal Band Positivity Using Routine Cerebrospinal Fluid and Serum Biochemical Markers"**, published in the *American Journal of Clinical Pathology (AJCP)*.

The project develops **SYNAPSI**, a weighted soft-voting ensemble model designed to predict Oligoclonal Band (OCB) positivity using cost-effective and routine laboratory parameters, offering a rapid screening tool for neuroinflammatory disorders.

## 📂 Repository Structure

The workflow is organized into sequential stages, indicated by the file prefixes:

* **`1_initial_model_screening.py`**:  
  Performs a comprehensive screening of multiple machine learning architectures (e.g., XGBoost, LightGBM, RF, SVM) using 5-fold cross-validation. It generates initial performance reports and SHAP summary plots to understand feature importance.

* **`2_candidate_evaluation.py`**:  
  Conducts a deeper evaluation of the top-performing candidate models selected from the screening phase. This step focuses on feature stability and hyperparameter sensitivity before the final ensemble construction.

* **`3_final_ensemble_model.py`**:  
  Constructs the final **SYNAPSI** model. This script performs hyperparameter optimization (via Optuna), trains the weighted soft-voting ensemble, and evaluates the final performance on the sequestered hold-out test set.

* **Helper Scripts**:
  * `helper_add_igg_index.py`: Preprocessing script to calculate IgG Index from raw values.
  * `helper_evaluate_igg_index.py`: Benchmarks the traditional IgG Index against the ML model.
  * `helper_statistical_comparison.py`: Performs statistical significance tests (DeLong's test) to compare ROC-AUC values.

## 🚀 Installation

Ensure you have Python 3.11+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt

💻 Usage
To reproduce the study pipeline, run the scripts in the following order. > Note: Please update the file paths inside the scripts (e.g., path/to/data.xlsx) to match your local directory structure.

Step 1: Data Preparation
Calculate the IgG Index (if not already present in your dataset):
python helper_add_igg_index.py

Step 2: Model Screening
Screen various algorithms to find the best baselines:
python 1_initial_model_screening.py

Step 3: Candidate Evaluation
Refine and analyze the top candidates:
python 2_candidate_evaluation.py

Step 4: Final Model (SYNAPSI)
Train, optimize, and validate the final ensemble model:
python 3_final_ensemble_model.py

Step 5: Statistical Comparison
Compare the model's performance against the conventional IgG Index:
python helper_statistical_comparison.py

📊 Data Availability
The clinical dataset used in this study contains sensitive patient information and cannot be made publicly available due to privacy regulations and ethical committee restrictions.

However, this repository provides the full methodological pipeline. Researchers can replicate the study by formatting their own private datasets to match the feature columns described in the code (e.g., CSF_Protein, Serum_IgG, Albumin_Quotient, etc.).

📝 Citation
If you use this code or methodology in your research, please cite our paper:

BibTeX:
@article{Gozgoz2025OCB,
  title={A machine learning model for predicting oligoclonal band positivity using routine cerebrospinal fluid and serum biochemical markers},
  author={Gözgöz, Hazar and Orhan, O. and Konuk, B. Akan and Akan, P.},
  journal={American Journal of Clinical Pathology},
  year={2025},
  publisher={Oxford University Press},
  doi={10.1093/ajcp/aqaf119} 
}

APA:

Gözgöz, H., Orhan, O., Konuk, B. A., & Akan, P. (2025). A machine learning model for predicting oligoclonal band positivity using routine cerebrospinal fluid and serum biochemical markers. American Journal of Clinical Pathology.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
