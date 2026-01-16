# ArXiv Reference Matching System

**Course:** NMKHDL - Lab 2  
**Student ID:** 23127088  
**Objective:** Automated matching of BibTeX entries with ArXiv metadata

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Installation Instructions](#installation-instructions)
5. [Execution Guide](#execution-guide)
6. [Output Files](#output-files)

---

## Project Overview

This system automatically matches BibTeX entries extracted from ArXiv LaTeX source files with their corresponding ArXiv metadata from `references.json` files. The pipeline uses hierarchical parsing, multi-stage data cleaning, feature engineering, and machine learning classification.

**Key Features:**
- Hierarchical LaTeX parsing (handles multi-file projects)
- Multi-tier data standardization (original → cleaned → no_stopwords)
- Feature engineering with 7 discriminative features
- Gradient Boosting classifier with hyperparameter optimization
- MRR@5 evaluation for ranking quality

**Performance:**
- **Parsing Success Rate:** 95.2%
- **Test MRR@5:** 0.8729
- **Perfect Match Rate:** 80% (rank 1)

---

## 🛠️ Environment Setup

### Prerequisites

- **Operating System:** Linux (tested on Ubuntu 20.04+) or WSL2
- **Python Version:** 3.10 or higher
- **Memory:** At least 4GB RAM
- **Storage:** At least 2GB free space

### Required Python Packages

The following packages are required (see `requirements.txt` for exact versions):

```txt
# Core Scientific Computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0

# Data Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Jupyter Support (optional, for notebooks)
jupyter>=1.0.0
ipykernel>=6.25.0

# Text Processing (standard library, no install needed)
# - re, json, pathlib, unicodedata, difflib
```

---

## Project Structure

```
Lab2/
├── README.md                          # This file
├── report.md                          # Detailed implementation report
├── requirements.txt                   # Python dependencies
│
├── papers/                            # ArXiv source files (input)
│   ├── 2312-15844/
│   │   ├── *.tex                      # LaTeX source files
│   │   ├── *.bib                      # Bibliography files (if any)
│   │   ├── metadata.json              # ArXiv metadata
│   │   └── references.json            # ArXiv references
│   ├── 2312-15845/
│   └── ...
│
├── bibtex/                            # Processed outputs (generated)
│   └── 23127088/                      # Student ID folder
│       ├── 2312-15844/
│       │   ├── refs.bib               # Extracted BibTeX entries
│       │   ├── cleaned_data.json      # Cleaned and standardized data
│       │   └── pred.json              # Model predictions (ranked list)
│       └── ...
│
├── labels/                            # Ground truth labels (generated)
│   └── ground_truth_labels.json      # Manual + automatic labels
│
├── features/                          # Feature datasets (generated)
│   ├── features_dataset.csv          # All samples with 7 features
│   ├── features_dataset.json         # JSON format
│   └── feature_metadata.json         # Feature statistics
│
├── models/                            # Trained models and results (generated)
│   ├── best_model.pkl                # Trained Gradient Boosting model
│   ├── scaler.pkl                    # Feature scaler
│   ├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
│   ├── model_metadata.json           # Model hyperparameters and metrics
│   ├── validation_results.csv        # Validation predictions
│   ├── test_predictions.csv          # Test predictions
│   ├── test_evaluation_mrr.csv       # Detailed MRR results
│   ├── confusion_matrix.png          # Confusion matrix plot
│   ├── roc_curve.png                 # ROC curve plot
│   ├── feature_importance_model.png  # Feature importance plot
│   ├── mrr_evaluation.png            # MRR comparison plot
│   └── rank_distribution_test.png    # Rank distribution plot
│
└── src/                               # Source code (Jupyter notebooks)
    ├── 2_1_Data_Cleaning.ipynb        # Step 1: Parse and clean data
    ├── 2_2_Data_Labelling.ipynb       # Step 2: Create ground truth labels
    ├── 2_3_Feature_Engineer.ipynb     # Step 3: Extract features
    ├── 2_4_Data_Modeling-Evaluation.ipynb  # Step 4: Train and evaluate model
    └── 3_3_Constructt_Submission_Folder.ipynb  # Step 5: Prepare submission
```

---

## Installation Instructions

### Step 1: Clone or Extract Project

```bash
# If using git
git clone <repository_url>
cd Lab2

# Or extract from archive
unzip Lab2.zip
cd Lab2
```

### Step 2: Create Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows (WSL2):
source venv/bin/activate
```

or using conda:

```bash
conda create -n lab2_env python=3.10 -y
conda activate lab2_env
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, pandas, numpy, matplotlib; print('All packages installed successfully')"
```

### Step 4: Verify Data Structure

Ensure the `papers/` directory contains ArXiv source files:

```bash
# Check if papers directory exists
ls -la papers/

# Should see directories like: 2312-15844, 2312-15845, etc.
# Each directory should contain: *.tex, references.json, metadata.json
```

---

## Execution Guide

### Quick Start (Run Full Pipeline)

Execute all notebooks in sequence:

```bash
# Navigate to src directory
cd src

# Run notebooks in order (if using command line)
jupyter nbconvert --execute --to notebook \
    --inplace 2_1_Data_Cleaning.ipynb \
    --ExecutePreprocessor.timeout=600

jupyter nbconvert --execute --to notebook \
    --inplace 2_2_Data_Labelling.ipynb \
    --ExecutePreprocessor.timeout=600

jupyter nbconvert --execute --to notebook \
    --inplace 2_3_Feature_Engineer.ipynb \
    --ExecutePreprocessor.timeout=600

jupyter nbconvert --execute --to notebook \
    --inplace 2_4_Data_Modeling-Evaluation.ipynb \
    --ExecutePreprocessor.timeout=600
```

**Or use Jupyter Notebook/Lab interface:**

```bash
# Start Jupyter
jupyter notebook

# Or Jupyter Lab
jupyter lab

# Then open and run each notebook in order
```

---

### Step-by-Step Execution

#### **Step 1: Data Cleaning and Parsing**

**Notebook:** `2_1_Data_Cleaning.ipynb`

**Purpose:** 
- Parse BibTeX entries from LaTeX source files
- Clean and standardize text (titles, authors, years)
- Create hierarchical representations

**Execution:**

```bash
# Open notebook
jupyter notebook 2_1_Data_Cleaning.ipynb

# Run all cells: Cell → Run All
# Or use: Shift+Enter on each cell
```

**Expected Runtime:** ~2-3 minutes

**Output:**
- `bibtex/23127088/<paper_id>/refs.bib` - Extracted BibTeX entries
- `bibtex/23127088/<paper_id>/cleaned_data.json` - Cleaned data with hierarchies

**Success Indicators:**
```
✓ Loaded X papers
✓ Processed X BibTeX entries
✓ Cleaning complete
✓ Saved cleaned data for X publications
```

---

#### **Step 2: Ground Truth Labeling**

**Notebook:** `2_2_Data_Labelling.ipynb`

**Purpose:**
- Create ground truth labels (manual + automatic)
- Match BibTeX entries to ArXiv IDs

**Execution:**

```bash
jupyter notebook 2_2_Data_Labelling.ipynb
# Run all cells
```

**Expected Runtime:** ~1 minute

**Output:**
- `labels/ground_truth_labels.json` - Manual and automatic labels

**Success Indicators:**
```
✓ Manual labels: 158
✓ Automatic labels: 0
✓ Total ground truth: 158
✓ Saved labels to labels/ground_truth_labels.json
```

---

#### **Step 3: Feature Engineering**

**Notebook:** `2_3_Feature_Engineer.ipynb`

**Purpose:**
- Generate (BibTeX, ArXiv) pairs
- Extract 7 features per pair
- Create balanced training dataset

**Execution:**

```bash
jupyter notebook 2_3_Feature_Engineer.ipynb
# Run all cells
```

**Expected Runtime:** ~1-2 minutes

**Output:**
- `features/features_dataset.csv` - Feature dataset (338 samples × 7 features)
- `features/features_dataset.json` - JSON format
- `features/feature_metadata.json` - Feature statistics

**Success Indicators:**
```
✓ Created 158 positive samples
✓ Created 180 negative samples
✓ Total samples: 338
✓ Features: 7
✓ Saved to features/features_dataset.csv
```

---

#### **Step 4: Model Training and Evaluation**

**Notebook:** `2_4_Data_Modeling-Evaluation.ipynb`

**Purpose:**
- Split data into train/valid/test sets
- Train Gradient Boosting classifier with Grid Search
- Evaluate using MRR@5 metric
- Generate predictions for all publications

**Execution:**

```bash
jupyter notebook 2_4_Data_Modeling-Evaluation.ipynb
# Run all cells
```

**Expected Runtime:** ~3-5 minutes (Grid Search may take longer)

**Output:**
- `models/best_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `models/model_metadata.json` - Hyperparameters and metrics
- `models/test_evaluation_mrr.csv` - Test results
- `bibtex/23127088/<paper_id>/pred.json` - Predictions for each paper

**Success Indicators:**
```
✓ Grid Search complete
✓ Best MRR@5 (Test): 0.8729
✓ Generated pred.json for 166 publications
✓ Perfect matches (Rank 1): 80%
✓ All models saved to models/
```

---

#### **Step 5: Construct Submission Folder** (Optional)

**Notebook:** `3_3_Constructt_Submission_Folder.ipynb`

**Purpose:**
- Copy metadata.json and references.json to submission folder
- Prepare final submission structure

**Execution:**

```bash
jupyter notebook 3_3_Constructt_Submission_Folder.ipynb
# Run all cells
```

**Expected Runtime:** <1 minute

**Output:**
- Copies files to `bibtex/23127088/<paper_id>/` for each publication

---

## Output Files

### Key Output Files

| File | Description | Usage |
|------|-------------|-------|
| `bibtex/23127088/<paper_id>/pred.json` | **Ranked predictions** for each BibTeX entry | Final output for submission |
| `models/model_metadata.json` | Model hyperparameters and performance | Reproducibility |
| `features/features_dataset.csv` | Feature dataset with labels | Analysis |
| `labels/ground_truth_labels.json` | Ground truth labels | Training/evaluation |

### Understanding pred.json

Each `pred.json` file contains:

```json
{
  "partition": "test",
  "groundtruth": {
    "smith2020": "2001.12345"
  },
  "prediction": {
    "smith2020": [
      "2001.12345",   // Rank 1 (correct match)
      "2002.54321",   // Rank 2
      "2003.98765",   // Rank 3
      "...",
      "..."
    ]
  }
}
```

- **partition:** train/valid/test
- **groundtruth:** BibTeX key → correct ArXiv ID
- **prediction:** BibTeX key → ranked list of ArXiv IDs (sorted by match probability)

---

### Performance Optimization

If execution is slow:

1. **Use fewer papers for testing:**
   ```python
   # In 2_1_Data_Cleaning.ipynb
   papers = papers[:10]  # Process only first 10 papers
   ```

2. **Reduce negative samples:**
   ```python
   # In 2_3_Feature_Engineer.ipynb
   negative_ratio = 1.0  # Instead of 2.0
   ```

3. **Skip Grid Search (use default parameters):**
   ```python
   # In 2_4_Data_Modeling-Evaluation.ipynb
   best_model = GradientBoostingClassifier(
       n_estimators=50, 
       max_depth=3, 
       learning_rate=0.1, 
       random_state=42
   )
   best_model.fit(X_train_scaled, y_train)
   ```

---

### Generated Files

```bash
# Check output files
ls -lh models/
# Should see: best_model.pkl, model_metadata.json, *.png, *.csv

ls -lh features/
# Should see: features_dataset.csv, feature_metadata.json

ls -lh bibtex/23127088/2312-15846/
# Should see: refs.bib, cleaned_data.json, pred.json
```