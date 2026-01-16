# arXiv Paper Scraper

A comprehensive tool for scraping arXiv papers, including metadata, LaTeX source files, and references using Arxiv API and Semantic Scholar API.

## Features

- Harvest metadata from arXiv using official API
- Download and extract LaTeX source files (all versions)
- Remove figure files to save space
- Extract references using Semantic Scholar API
- Checkpoint system for resuming interrupted scraping
- Comprehensive statistics tracking
- Organized file structure

## Installation

### Local Setup

1. Download this repository

2. Go to the src directory:
```bash
cd src
```

3. Install dependencies:
```bash
python -m venv env
call env/Scripts/activate.bat  # On Windows
pip install -r requirements.txt
```
or using miniconda:
```bash
conda create --name env python
conda activate env
pip install -r requirements.txt
```

4. Run the scraper:
```bash
python main.py
```

### Google Colab Setup

1. Open a new Google Colab notebook
2. Upload the notebook file `main.ipynb`
3. Run all cells sequentially
4. Download results when complete

## Configuration

Edit `config.py` to customize:

### arXiv ID Ranges
```python
ARXIV_ID_RANGES = [
    ("2312", 15844, 99999),  # December 2023
    ("2401", 0, 3095),        # January 2024
]
```

### API Settings
```python
ARXIV_DELAY = 3.0              # Delay between arXiv API calls
ARXIV_BATCH_SIZE = 100         # Papers per batch
REQUEST_DELAY = 3              # Delay for other APIs
```

### Download Settings
```python
DOWNLOAD_BATCH_SIZE = 5        # Concurrent downloads
STATS_SAVE_INTERVAL = 50       # Save checkpoint every N papers
```