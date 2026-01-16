from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
STATS_DIR = DATA_DIR / "statistics"

# Create directories if they don't exist
for directory in [PAPERS_DIR, TEMP_DIR, LOGS_DIR, STATS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# arXiv settings
ARXIV_ID_RANGES = [
    ("2312", 15844, 15848),
    # ("2312", 15844, 16344),
    # ("2312", 15844, 99999),
    # ("2401", 0, 3095),
]

# API settings
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
REQUEST_DELAY = 2  # Delay between reference API calls
MAX_RETRIES = 5
RETRY_DELAY = 3.0


# arXiv API settings
ARXIV_MAX_RESULTS = 100
ARXIV_DELAY = 3.0
ARXIV_BATCH_SIZE = 100

# Download settings
PAPER_DOWNLOAD_BATCH_SIZE = 5  # Concurrent papers to download (main parallel)
REFERENCE_BATCH_SIZE = 2  # Concurrent reference API calls

# Figure extensions to remove
FIGURE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.ps', '.gif', '.svg', '.bmp']

# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024

# Stats settings
STATS_SAVE_INTERVAL = 100

# Cleanup settings
DELETE_RAW_AFTER_EXTRACT = True
DELETE_TEMP_AFTER_COPY = True




# Semantic Scholar fields
SEMANTIC_SCHOLAR_FIELDS = "references,references.title,references.authors,references.year,references.paperId,references.externalIds"
SEMANTIC_KEY_API = "FkKyqq5p9R96KizMlPWBwa0YtPjCvyq24ddno82O"