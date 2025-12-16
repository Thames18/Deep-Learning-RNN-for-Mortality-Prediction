# Deep Learning RNN for Mortality Prediction

This project implements a **recurrent neural network (RNN)**–based pipeline for **in-hospital mortality prediction** using structured clinical time-series data. It was developed as part of **CSE6250: Big Data Analytics in Healthcare** at **Georgia Institute of Technology**.

The project focuses on **ETL preprocessing**, **variable-length sequence modeling**, and **deep learning with PyTorch** in a reproducible, test-driven environment.

---

## Project Overview

Predicting patient mortality from longitudinal electronic health record (EHR) data is a core problem in healthcare analytics. In this project, we build an end-to-end deep learning pipeline that:

- Transforms raw clinical data into model-ready features  
- Handles **variable-length patient sequences**  
- Trains an RNN-based classifier using **PyTorch**  
- Evaluates model performance using appropriate classification metrics  

---

## Key Features

- End-to-end ETL pipeline for mortality prediction data  
- Custom PyTorch `Dataset` and `DataLoader` for sequential clinical data  
- Variable-length RNN model implementation  
- Reproducible training and evaluation scripts  
- Automated unit tests to ensure correctness  

---

## Project Structure
homework4/
├── code/
│ ├── etl_mortality_data.py # Data preprocessing and feature engineering
│ ├── mydatasets.py # Custom PyTorch Dataset / DataLoader
│ ├── mymodels.py # RNN model definitions
│ ├── train_variable_rnn.py # Training and evaluation script
│ ├── utils.py # Helper functions
│ ├── plots.py # Visualization utilities
│ ├── tests/
│ │ └── test_all.py # Unit tests
├── data/
│ └── mortality/ # Clinical mortality dataset
├── environment.yml # Conda environment configuration
└── README.md---

## Technologies Used

- **Programming Language:** Python 3.6–3.8  
- **Deep Learning:** PyTorch  
- **Data Processing:** NumPy, Pandas, SciPy  
- **Machine Learning:** Scikit-learn  
- **Environment Management:** Conda  
- **Testing:** PyTest / custom unit tests  

---

## Model Description

- Recurrent Neural Network (RNN) for binary mortality classification  
- Handles **variable-length clinical time-series**  
- Processes structured EHR features derived from admissions data  
- Trained using supervised learning with cross-entropy loss  

> Note: Model architecture and training setup follow the assignment specification provided in CSE6250 HW4.

---

## Setup Instructions

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate homework4

### 2. Preprocess Data
python code/etl_mortality_data.py

3. Train the Model
python code/train_variable_rnn.py

4. Run Tests
pytest code/tests/

Evaluation

Model performance is evaluated using classification metrics suitable for imbalanced clinical outcomes, including:

Area Under the Precision–Recall Curve (AUPRC)

ROC-AUC

Accuracy

Academic Integrity

This repository contains original work completed for CSE6250 in accordance with the Georgia Tech Honor Code.
Discussion was permitted, but all code and written solutions are my own.

Note: Due to data usage restrictions, the dataset is not included in this repository.

Author:
Mustafa Al-Salem
Master of Computer Science — Georgia Institute of Technology

Disclaimer

This project is for educational purposes only and is not intended for clinical use.
