# ArXiv Paper Scraping and Reference Matching System

**Course:** NMKHDL  
**Project:** Automated ArXiv paper collection and BibTeX reference matching

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Installation Instructions](#installation-instructions)
5. [Execution Guide](#execution-guide)
6. [Output Files](#output-files)

---

## Project Overview

This project consists of two integrated systems for working with ArXiv academic papers:

### Part 1: Scraping System
A comprehensive tool for collecting ArXiv papers, including:
- Harvesting metadata from ArXiv using official API
- Downloading and extracting LaTeX source files (all versions)
- Extracting references using Semantic Scholar API
- Checkpoint system for resuming interrupted scraping
- Comprehensive statistics tracking

### Part 2: Modeling System
An automated matching system that:
- Parses BibTeX entries from LaTeX source files
- Cleans and standardizes bibliographic data
- Engineers discriminative features
- Uses machine learning to match BibTeX entries with ArXiv metadata
- Evaluates matching quality using MRR@5 metrics

**Key Performance Metrics:**
- Parsing Success Rate: 95.2%
- Test MRR@5: 0.8729
- Perfect Match Rate: 80% (rank 1)

---

## Project Structure

```
Arxiv-Scraping-Modeling/
│
├── Scraping/                          # Paper collection system
│   ├── README.md                      # Scraping documentation
│   ├── environment.yaml               # Conda environment file
│   └── src/
│       ├── main.py                    # Main scraper script
│       ├── main.ipynb                 # Jupyter notebook version
│       ├── config.py                  # Configuration settings
│       ├── requirements.txt           # Python dependencies
│       ├── lib/                       # Core scraping modules
│       │   ├── metadata_updater.py
│       │   ├── source_downloader.py
│       │   ├── source_identifier.py
│       │   ├── reference_extractor.py
│       │   ├── paper_organizer.py
│       │   └── statistics_tracker.py
│       ├── data/                      # Scraped data output
│       │   ├── papers/                # Downloaded papers
│       │   ├── references/            # Extracted references
│       │   ├── metadata/              # Paper metadata
│       │   └── statistics/            # Scraping statistics
│       └── logs/                      # Execution logs
│
└── Modeling/                          # Reference matching system
    ├── README.md                      # Modeling documentation
    ├── report.md                      # Detailed implementation report
    ├── requirements.txt               # Python dependencies
    ├── src/                           # Source code (Jupyter notebooks)
    │   ├── 2_1_Data_Cleaning.ipynb
    │   ├── 2_2_Data_Labelling.ipynb
    │   ├── 2_3_Feature_Engineer.ipynb
    │   ├── 2_4_Data_Modeling-Evaluation.ipynb
    │   └── 3_3_Constructt_Submission_Folder.ipynb
    ├── papers/                        # ArXiv source files (input)
    │   ├── 2312-15844/
    │   │   ├── *.tex
    │   │   ├── *.bib
    │   │   ├── metadata.json
    │   │   └── references.json
    │   └── ...
    ├── bibtex/                        # Processed outputs
    │   └── 23127088/
    │       ├── <paper_id>/
    │       │   ├── refs.bib
    │       │   ├── cleaned_data.json
    │       │   └── pred.json
    │       └── ...
    ├── labels/                        # Ground truth labels
    │   └── ground_truth_labels.json
    ├── features/                      # Feature datasets
    │   ├── features_dataset.csv
    │   ├── features_dataset.json
    │   └── feature_metadata.json
    └── models/                        # Trained models and results
        ├── best_model.pkl
        ├── scaler.pkl
        ├── tfidf_vectorizer.pkl
        ├── model_metadata.json
        ├── validation_results.csv
        ├── test_predictions.csv
        └── test_evaluation_mrr.csv
```

---

## Environment Setup

### Prerequisites

- **Operating System:** Linux, macOS, Windows (WSL2 recommended for Windows)
- **Python Version:** 3.10 or higher
- **Memory:** At least 4GB RAM
- **Storage:** At least 5GB free space
- **Internet:** Required for API access

### Required Python Packages

#### For Scraping System:
```txt
arxiv
requests
psutil
```

#### For Modeling System:
```txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
tqdm>=4.65.0
```

---

## Installation Instructions

### Option 1: Using Virtual Environment (Recommended)

#### Step 1: Create Virtual Environment

```bash
# Navigate to project root
cd Arxiv-Scraping-Modeling

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# On WSL2:
source venv/bin/activate
```

#### Step 2: Install Scraping Dependencies

```bash
cd Scraping/src
pip install --upgrade pip
pip install -r requirements.txt
cd ../..
```

#### Step 3: Install Modeling Dependencies

```bash
cd Modeling
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

#### Step 4: Verify Installation

```bash
# Verify scraping dependencies
python -c "import arxiv, requests, psutil; print('Scraping dependencies OK')"

# Verify modeling dependencies
python -c "import sklearn, pandas, numpy, matplotlib; print('Modeling dependencies OK')"
```

---

### Option 2: Using Conda (Alternative)

#### For Scraping System:

```bash
cd Scraping
conda env create -f environment.yaml
conda activate <env_name>
cd ..
```

Or manually:

```bash
conda create -n scraping_env python=3.10 -y
conda activate scraping_env
cd Scraping/src
pip install -r requirements.txt
cd ../..
```

#### For Modeling System:

```bash
conda create -n modeling_env python=3.10 -y
conda activate modeling_env
cd Modeling
pip install -r requirements.txt
cd ..
```

---

## Execution Guide

### Part 1: Running the Scraping System

#### Local Execution

```bash
# Navigate to scraping source directory
cd Scraping/src

# Activate environment (if using venv)
source ../../venv/bin/activate  # Linux/Mac
# or
..\..\venv\Scripts\activate  # Windows

# Run the scraper
python main.py
```

#### Configuration

Edit `Scraping/src/config.py` to customize scraping parameters:

```python
# ArXiv ID ranges to scrape
ARXIV_ID_RANGES = [
    ("2312", 15844, 99999),  # December 2023
    ("2401", 0, 3095),        # January 2024
]

# API delays (to respect rate limits)
ARXIV_DELAY = 3.0              # Delay between arXiv API calls
ARXIV_BATCH_SIZE = 100         # Papers per batch
REQUEST_DELAY = 3              # Delay for other APIs

# Download settings
DOWNLOAD_BATCH_SIZE = 5        # Concurrent downloads
STATS_SAVE_INTERVAL = 50       # Save checkpoint every N papers
```

#### Using Jupyter Notebook

```bash
cd Scraping/src
jupyter notebook main.ipynb
# Run all cells sequentially
```

#### Google Colab Setup

1. Open Google Colab
2. Upload `Scraping/src/main.ipynb`
3. Run all cells sequentially
4. Download results when complete

---

### Part 2: Running the Modeling System

#### Step-by-Step Execution

The modeling pipeline consists of 5 sequential notebooks:

```bash
# Navigate to modeling source directory
cd Modeling/src

# Activate environment
source ../../venv/bin/activate  # Linux/Mac
# or
..\..\venv\Scripts\activate  # Windows

# Start Jupyter
jupyter notebook
# or
jupyter lab
```

Then execute notebooks in order:

#### **Notebook 1: Data Cleaning and Parsing** (`2_1_Data_Cleaning.ipynb`)

**Purpose:** Parse BibTeX entries from LaTeX source files and standardize data

**Steps:**
1. Open `2_1_Data_Cleaning.ipynb`
2. Run all cells (Cell → Run All or Shift+Enter)
3. Expected runtime: ~2-3 minutes

**Output:**
- `bibtex/23127088/<paper_id>/refs.bib` - Extracted BibTeX entries
- `bibtex/23127088/<paper_id>/cleaned_data.json` - Cleaned data

---

#### **Notebook 2: Ground Truth Labeling** (`2_2_Data_Labelling.ipynb`)

**Purpose:** Create ground truth labels for model training

**Steps:**
1. Open `2_2_Data_Labelling.ipynb`
2. Run all cells
3. Expected runtime: ~1 minute

**Output:**
- `labels/ground_truth_labels.json` - Manual and automatic labels

---

#### **Notebook 3: Feature Engineering** (`2_3_Feature_Engineer.ipynb`)

**Purpose:** Extract discriminative features for matching

**Steps:**
1. Open `2_3_Feature_Engineer.ipynb`
2. Run all cells
3. Expected runtime: ~5-10 minutes

**Output:**
- `features/features_dataset.csv` - Feature matrix
- `features/features_dataset.json` - Feature data in JSON
- `features/feature_metadata.json` - Feature statistics

**Features Extracted:**
- Levenshtein distance similarity
- Fuzzy matching scores
- TF-IDF cosine similarity
- Year matching
- Author overlap
- Title word overlap
- Combined text similarity

---

#### **Notebook 4: Model Training and Evaluation** (`2_4_Data_Modeling-Evaluation.ipynb`)

**Purpose:** Train Gradient Boosting classifier and evaluate performance

**Steps:**
1. Open `2_4_Data_Modeling-Evaluation.ipynb`
2. Run all cells
3. Expected runtime: ~10-15 minutes

**Output:**
- `models/best_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `models/model_metadata.json` - Model hyperparameters
- `models/validation_results.csv` - Validation predictions
- `models/test_predictions.csv` - Test predictions
- `models/test_evaluation_mrr.csv` - MRR evaluation results
- Various plots (confusion matrix, ROC curve, feature importance, etc.)

---

#### **Notebook 5: Construct Submission** (`3_3_Constructt_Submission_Folder.ipynb`)

**Purpose:** Prepare final submission folder with predictions

**Steps:**
1. Open `3_3_Constructt_Submission_Folder.ipynb`
2. Run all cells
3. Expected runtime: ~2-3 minutes

**Output:**
- `bibtex/23127088/<paper_id>/pred.json` - Ranked predictions for each paper

---

### Quick Start (Command Line Execution)

For automated execution of all modeling notebooks:

```bash
cd Modeling/src

# Execute all notebooks in sequence
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

jupyter nbconvert --execute --to notebook \
    --inplace 3_3_Constructt_Submission_Folder.ipynb \
    --ExecutePreprocessor.timeout=600
```

---

## Output Files

### Scraping System Outputs

Located in `Scraping/src/data/`:

```
data/
├── papers/                    # Downloaded ArXiv papers
│   └── <arxiv_id>/
│       ├── metadata.json      # Paper metadata
│       ├── references.json    # Extracted references
│       └── tex/               # LaTeX source files
│           └── <version>/
│               ├── *.tex
│               └── *.bib
├── references/                # Reference data
│   └── <arxiv_id>_references.json
├── metadata/                  # Metadata cache
│   └── paper_list.json
└── statistics/                # Scraping statistics
    └── checkpoint_latest.json
```

### Modeling System Outputs

Located in `Modeling/`:

```
Modeling/
├── bibtex/23127088/           # Per-paper outputs
│   └── <paper_id>/
│       ├── refs.bib           # Extracted BibTeX entries
│       ├── cleaned_data.json  # Cleaned and standardized data
│       └── pred.json          # Model predictions (ranked)
├── labels/
│   └── ground_truth_labels.json  # Training labels
├── features/
│   ├── features_dataset.csv   # Feature matrix
│   ├── features_dataset.json  # Feature data (JSON)
│   └── feature_metadata.json  # Feature statistics
└── models/
    ├── best_model.pkl         # Trained Gradient Boosting model
    ├── scaler.pkl             # Feature scaler
    ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
    ├── model_metadata.json    # Model hyperparameters and metrics
    ├── validation_results.csv # Validation predictions
    ├── test_predictions.csv   # Test set predictions
    ├── test_evaluation_mrr.csv # MRR evaluation results
    └── *.png                  # Visualization plots
```

---

## Workflow

### Complete Pipeline

1. **Scrape ArXiv Papers:**
   ```bash
   cd Scraping/src
   python main.py
   ```
   Collects papers with metadata, LaTeX sources, and references.

2. **Transfer Data to Modeling:**
   ```bash
   # Copy scraped papers to Modeling/papers/
   cp -r Scraping/src/data/papers/* Modeling/papers/
   ```

3. **Run Modeling Pipeline:**
   ```bash
   cd Modeling/src
   jupyter notebook
   # Execute notebooks 1-5 in sequence
   ```

4. **Review Results:**
   - Check `Modeling/models/test_evaluation_mrr.csv` for performance metrics
   - Review `Modeling/bibtex/23127088/<paper_id>/pred.json` for predictions
   - Examine visualization plots in `Modeling/models/`

---

## Troubleshooting

### Common Issues

**Scraping System:**

1. **API Rate Limiting:**
   - Increase `ARXIV_DELAY` and `REQUEST_DELAY` in `config.py`
   - Default delays are conservative (3 seconds)

2. **Download Failures:**
   - Check internet connection
   - Verify ArXiv IDs are valid
   - Review logs in `Scraping/src/logs/`

3. **Checkpoint Recovery:**
   - Script automatically resumes from last checkpoint
   - Check `Scraping/src/data/statistics/checkpoint_latest.json`

**Modeling System:**

1. **Missing Dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Memory Issues:**
   - Reduce batch sizes in notebook cells
   - Close other applications
   - Use a machine with more RAM

3. **File Not Found Errors:**
   - Ensure `papers/` directory has scraped data
   - Verify directory structure matches expected layout
   - Check file paths in notebook cells

4. **Kernel Crashes:**
   - Restart Jupyter kernel
   - Clear outputs and run again
   - Check system resources

---

## Additional Documentation

- **Scraping System:** See `Scraping/README.md` for detailed scraping documentation
- **Modeling System:** See `Modeling/README.md` for detailed modeling documentation
- **Implementation Report:** See `Modeling/report.md` for technical details and analysis
- **Manual Labeling Guide:** See `Modeling/MANUAL_LABELING_GUIDE.md` for labeling instructions

---

## Notes

- The scraping system respects API rate limits with configurable delays
- Checkpoint system ensures scraping can be resumed after interruptions
- The modeling system uses hierarchical parsing to handle complex LaTeX projects
- Feature engineering includes 7 discriminative features for robust matching
- Gradient Boosting classifier with hyperparameter optimization achieves 87.29% MRR@5
- All file paths use forward slashes for cross-platform compatibility

---