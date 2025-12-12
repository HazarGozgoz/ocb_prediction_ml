# OCB Prediction ML: A Machine Learning Pipeline for CSF Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Published in AJCP](https://img.shields.io/badge/Published%20in-AJCP-red)]([https://academic.oup.com/ajcp](https://doi.org/10.1093/ajcp/aqaf119))

## Overview
This repository contains the official implementation of the machine learning pipeline described in the paper: **"A Machine Learning Model for Predicting Oligoclonal Band Positivity Using Routine Cerebrospinal Fluid and Serum Biochemical Markers"**, published in the *American Journal of Clinical Pathology (AJCP)*.

The project develops **SYNAPSI**, a weighted soft-voting ensemble model designed to predict Oligoclonal Band (OCB) positivity using cost-effective and routine laboratory parameters, offering a rapid screening tool for neuroinflammatory disorders.

## 📂 Repository Structure

The workflow is organized into sequential stages, indicated by the file prefixes:

* **`1_initial_model_screening.py`**:  
    Performs a comprehensive screening of multiple machine learning architectures (e.g., XGBoost, LightGBM, RF, SVM) using 5-fold cross-validation to identify candidate models. Generates initial performance reports and SHAP summary plots.

* **`2_candidate_evaluation.py`**:  
    Conducts a deeper evaluation of the top-performing candidate models selected from step 1. Focuses on feature stability and hyperparameter sensitivity before the final ensemble construction.

* **`3_final_ensemble_model.py`**:  
    Constructs the final **SYNAPSI** model. This script performs hyperparameter optimization (via Optuna), trains the weighted soft-voting ensemble, and evaluates the final performance on the sequestered hold-out test set.

* **Helper Scripts**:
    * `helper_add_igg_index.py`: Preprocessing script to calculate IgG Index from raw values.
    * `helper_evaluate_igg_index.py`: Benchmarks the traditional IgG Index against the ML model.
    * `helper_statistical_comparison.py`: Performs statistical significance tests (DeLong's test, etc.) to compare ROC-AUC values.

## 🚀 Installation

Ensure you have Python 3.11+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
